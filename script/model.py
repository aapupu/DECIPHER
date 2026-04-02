import torch
import torch.nn as nn
import torch.nn.functional as F 

import os
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

from loss import LossWeights, compute_losses
from utils import EarlyStopping

@dataclass
class ForwardOut:
    z_c: torch.Tensor
    z_s: torch.Tensor
    pi: torch.Tensor
    p: torch.Tensor
    z_hat: torch.Tensor
    recon_x: torch.Tensor
    domain_logits: torch.Tensor

class DECIPHER(nn.Module):
    def __init__(self, s_z, n_feature=2000, hidden_dim=(128,128), 
                 n_domain=2, drop_prob=0.1, 
                 batch_norm=False, layer_norm=True, activation='leaky_relu'):
        super().__init__()
        # ---- normalize and flatten prototypes ----
        # Accept s_z as:
        #   (M, dc) or (C, K, dc)
        if s_z.dim() == 3:
            C, K, dc = s_z.shape
            S = s_z.reshape(C * K, dc)
            proto2ct = torch.arange(C).repeat_interleave(K)
            self.num_celltypes = C
            self.num_prototypes = C * K
        elif s_z.dim() == 2:
            # Assume prototypes already flattened; need user to provide mapping?
            # Here we treat each prototype as a "celltype" if mapping unknown.
            M, dc = s_z.shape·
            S = s_z
            proto2ct = torch.arange(M)
            self.num_celltypes = M
            self.num_prototypes = M
        else:
            raise ValueError("s_z must be 2D (M, dc) or 3D (C, K, dc).")

        # L2 normalize each prototype row for NNLS stability
        S = S / (S.norm(dim=-1, keepdim=True) + 1e-8)

        self.register_buffer("S_z", S)
        self.register_buffer("proto2ct", proto2ct.long())

        latent_dim = S.size(-1)
        self.latent_dim = latent_dim

        if len(s_z.shape) == 3:
            self.n_cell_state = s_z.shape[1]
        elif len(s_z.shape) == 2:
            self.n_cell_state = 0

        self.decoupled_encoder = DecoupledEncoder(latent_dim, n_feature, hidden_dim, drop_prob, 
                                                  batch_norm, layer_norm, activation)
        
        decoder_hidden_dim = hidden_dim if isinstance(hidden_dim, (int, float)) else hidden_dim[-1]
        self.decoder = Decoder(latent_dim, n_feature, decoder_hidden_dim, drop_prob, 
                                batch_norm, layer_norm, activation='leaky_relu')

        self.domain_classifier = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(latent_dim//2, n_domain)
            )
        self.solver = MirrorDescentNNLS(steps=20, init_eta=0.5, sum_to_one=False,
                                        learn_eta=True, eps=1e-8)

    def forward(self, x):
        z_c, z_s = self.decoupled_encoder(x)

        pi, p, z_hat = self.solve_pi(z_c)
        recon_x = self.decoder(z_c, z_s)

        domain_logits = self.domain_classify(z_s)
        return ForwardOut(
            z_c=z_c,
            z_s=z_s,
            pi=pi, #[B, C * K]
            p=p, #[B, C]
            z_hat=z_hat,
            recon_x=recon_x,
            domain_logits=domain_logits,
        )

    def solve_pi(self, z_c: torch.Tensor):
        """
        Return:
          pi: [B, M]
          p:  [B, C]
          z_hat: [B, dc]
        """
        z_c = z_c / (z_c.norm(dim=-1, keepdim=True) + 1e-8)
        pi = self.solver(z_c, self.S_z)   # [B, M]
        z_hat = pi @ self.S_z             # [B, dc]

        # aggregate -> p
        B = pi.size(0)
        C = int(self.num_celltypes)
        p = torch.zeros(B, C, device=pi.device, dtype=pi.dtype)
        p.scatter_add_(1, self.proto2ct.view(1, -1).expand(B, -1), pi)
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
        return pi, p, z_hat

    def domain_classify(self, z_s):
        z_s_dom = z_s[:, :self.latent_dim//2]
        domain_logits = self.domain_classifier(z_s_dom)
        return domain_logits

    def fit(self, train_dataloader, 
             lr=1e-4, weight_decay=1e-3,
            max_epoch=500,  device='cuda', patience=10,
            loss_weight=LossWeights(),
            outdir=None, verbose=False,
            logger=None):
        self.to(device)
        self.train()

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        early_stopping = EarlyStopping(patience=patience, verbose=verbose,
                path=outdir if outdir else os.path.join(os.getcwd(), 'DECIPHER.pt'))
        
        track_keys = ["L_total", "L_prop", "L_rec", "L_latrec", "L_dom", "L_decouple", "L_align"]
        hist = {k: [] for k in track_keys}
        t = tqdm(range(max_epoch), desc="Epochs")
        for epoch in t:
            for idx, (pesudo, real, _) in enumerate(train_dataloader):
                x_pesudo = pesudo['x'].float().to(device)
                domain_pesudo = pesudo['domain'].long().to(device)
                prop_pesudo = pesudo['prop'].float().to(device)

                x_real = real['x'].float().to(device)
                domain_real =real['domain'].long().to(device)

                out_pseudo = self(x_pesudo)
                out_real = self(x_real)

                losses = compute_losses(out_pseudo, out_real, x_pesudo, x_real, prop_pesudo,
                               domain_pesudo, domain_real, loss_weight)
                total = losses["L_total"]

                optimizer.zero_grad()
                total.backward()
                optimizer.step()

                # record
                vals = {
                    "L_total": total,
                    "L_prop": losses["L_prop"],
                    "L_rec": losses["L_rec"],
                    "L_latrec": losses["L_latrec"],
                    "L_dom": losses["L_dom"],
                    "L_decouple": losses["L_decouple"],
                    "L_align": losses["L_align"],
                }
                for k, v in vals.items():
                    hist[k].append(v.detach().item())

            # epoch summary
            avg = {k: float(np.mean(v)) for k, v in hist.items()}
            info_str = (
                f"total={avg['L_total']:.2f} "
                f"prop={avg['L_prop']:.2f} rec={avg['L_rec']:.2f} latrec={avg['L_latrec']:.2f} "
                f"dom={avg['L_dom']:.2f} dec={avg['L_decouple']:.2f} align={avg['L_align']:.2f}"
            )
            if logger:
                logger.info(info_str)
            else:
                t.set_postfix_str(info_str)

            early_stopping(avg['L_total'], self)
            if early_stopping.early_stop:
                if logger: 
                    logger.info(f"EarlyStopping: run {epoch+1} epoch")
                else:
                    print(f"EarlyStopping: run {epoch+1} epoch")
                break

    @torch.no_grad()  
    def encode_z(self, x):
        self.eval()
        return self.decoupled_encoder(x)
    
    @torch.no_grad()
    def deconvolution(self, x: torch.Tensor):
        self.eval()
        zc, _ = self.decoupled_encoder(x)
        pi, p, _ = self.solve_pi(zc)
        return pi, p

class MirrorDescentNNLS(nn.Module):
    """
    Solve:
        min_{pi >= 0} || z - pi @ S ||^2  (+ optional normalization to sum-to-one)

    If sum_to_one=True, we renormalize pi each iteration: pi <- pi / sum(pi).
    This gives simplex-like behavior and stabilizes proportions.
    """
    def __init__(
        self,
        steps: int = 20,
        init_eta: float = 0.5,
        sum_to_one: bool = False,
        learn_eta: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.steps = int(steps)
        self.sum_to_one = sum_to_one
        self.eps = eps
        eta = torch.tensor(float(init_eta))
        self.eta = nn.Parameter(eta) if learn_eta else eta

    def forward(self, z: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        z: [B, dc]
        S: [M, dc]
        returns pi: [B, M], non-negative; if sum_to_one -> rows sum to 1
        """
        B = z.size(0)
        M = S.size(0)
        pi = torch.full((B, M), 1.0 / M, device=z.device, dtype=z.dtype)

        eta = self.eta.clamp(min=1e-4, max=10.0)

        for _ in range(self.steps):
            z_hat = pi @ S                      # [B, dc]
            grad = 2.0 * (z_hat - z) @ S.t()    # [B, M]
            pi = pi * torch.exp(-eta * grad).clamp(min=self.eps)

            if self.sum_to_one:
                pi = pi / (pi.sum(dim=-1, keepdim=True) + self.eps)

        return pi
    
class DecoupledEncoder(nn.Module):
    def __init__(self, latent_dim, n_feature, hidden_dim, drop_prob, 
                 batch_norm, layer_norm, activation):
        super().__init__()
        self.common_encoder = MultiDenseLayer(n_feature, hidden_dim, drop_prob, 
                                        batch_norm, layer_norm, activation)
        last_hidden_dim = hidden_dim if isinstance(hidden_dim, (int, float)) else hidden_dim[-1]
        self.z_c_dense = nn.Sequential(
            nn.Linear(last_hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim, elementwise_affine=False)
        )
        self.z_s_dense = nn.Sequential(
            nn.Linear(last_hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim, elementwise_affine=False)
        )

    def forward(self, x):
        h = self.common_encoder(x)
        z_c = self.z_c_dense(h)
        z_s = self.z_s_dense(h)
        return z_c, z_s

class Decoder(nn.Module):
    def __init__(self, latent_dim, n_feature, hidden_dim, drop_prob, 
                 batch_norm=False, layer_norm=True, 
                 activation='leaky_relu'):
        super().__init__()
        self.dense1 = MultiDenseLayer(latent_dim*2, hidden_dim, drop_prob, 
                                     batch_norm, layer_norm, activation)
        last_hidden_dim = hidden_dim if isinstance(hidden_dim, (int, float)) else hidden_dim[-1]
        self.dense2 = nn.Sequential(
            nn.Linear(last_hidden_dim, n_feature),
            ActivationFactory.get('relu')
        )

    def forward(self, z_c, z_s):
        h = self.dense1(torch.cat([z_c, z_s], dim=-1))
        recon_x = self.dense2(h)
        return recon_x

class DenseLayer(nn.Module):
    def __init__(self, in_dim, out_dim, drop_prob=0.0, batch_norm=False, layer_norm=False, activation='relu'):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim)]

        if batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(out_dim, elementwise_affine=False))

        layers.append(ActivationFactory.get(activation))

        if drop_prob > 0:
            layers.append(nn.Dropout(drop_prob))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
    
class MultiDenseLayer(nn.Module):
    def __init__(self, input_dim, hidden, drop_prob, 
                 batch_norm=True, layer_norm=False, 
                 activation='leaky_relu'):
        super().__init__()
        layers = []
        if isinstance(hidden, (int, float)):
            layers.append(DenseLayer(input_dim, hidden, drop_prob,
                                     batch_norm=batch_norm, 
                                     layer_norm=layer_norm,
                                     activation=activation))
        elif isinstance(hidden, (list, tuple)) and len(hidden) >= 1:
            in_dim = input_dim
            for h_dim in hidden:
                layers.append(DenseLayer(in_dim, h_dim, drop_prob, 
                                         batch_norm=batch_norm, 
                                         layer_norm=layer_norm,
                                         activation=activation))
                in_dim = h_dim
        else:
            raise ValueError("`hidden` must be an int, float, list, or tuple with at least one element.")

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class ActivationFactory:
    """Factory class to create activation functions by name."""
    
    _activations = {
        'relu': lambda: nn.ReLU(inplace=True),
        'leaky_relu': lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True),
        'gelu': lambda: nn.GELU(),
        'tanh': lambda: nn.Tanh(),
        'sigmoid': lambda: nn.Sigmoid(),
        'softmax': lambda: nn.Softmax(-1),
        'none': lambda: nn.Identity(),
    }

    @staticmethod
    def get(name: str) -> nn.Module:
        if name not in ActivationFactory._activations:
            raise ValueError(f"Unsupported activation: {name}")
        return ActivationFactory._activations[name]()
