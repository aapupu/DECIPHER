import os
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from .loss import LossWeights, compute_losses
from .utils import EarlyStopping


@dataclass
class ForwardOut:
    z_c: torch.Tensor
    z_s: torch.Tensor
    pi: torch.Tensor
    p: torch.Tensor
    z_hat: torch.Tensor
    recon_x: torch.Tensor
    # z_s domain-supervised head.
    domain_logits_s: torch.Tensor


class DECIPHER(nn.Module):
    def __init__(
        self,
        s_z,
        n_feature=2000,
        hidden_dim=(128, 128),
        proto2ct=None,
        n_celltypes=None,
        n_domain=2,
        drop_prob=0.1,
        batch_norm=False,
        layer_norm=True,
        activation='leaky_relu',
        align_dim=32,
    ):
        super().__init__()
        # Prepare prototypes
        if s_z.dim() == 3:
            C, K, dc = s_z.shape
            S = s_z.reshape(C * K, dc)
            if proto2ct is None:
                proto2ct = torch.arange(C).repeat_interleave(K)
            else:
                proto2ct = torch.as_tensor(proto2ct).long()
                assert proto2ct.numel() == C * K, "proto2ct length must equal C*K"
            self.num_celltypes = int(proto2ct.max().item()) + 1 if n_celltypes is None else int(n_celltypes)
            self.num_prototypes = C * K
            self.n_cell_state = K
        elif s_z.dim() == 2:
            M, dc = s_z.shape
            S = s_z
            if proto2ct is None:
                proto2ct = torch.arange(M)
                self.num_celltypes = M
            else:
                proto2ct = torch.as_tensor(proto2ct).long()
                assert proto2ct.numel() == M, "proto2ct length must equal number of prototypes M"
                self.num_celltypes = int(proto2ct.max().item()) + 1 if n_celltypes is None else int(n_celltypes)
            self.num_prototypes = M
            self.n_cell_state = 0
        else:
            raise ValueError("s_z must be 2D (M, dc) or 3D (C, K, dc).")

        S = S / (S.norm(dim=-1, keepdim=True) + 1e-8)
        self.register_buffer("S_z_raw", S)
        self.register_buffer("proto2ct", proto2ct.long())

        self.latent_dim = int(align_dim)
        self.n_domain = int(n_domain)
        self.proto_projector = nn.Sequential(
            nn.Linear(dc, self.latent_dim),
            ActivationFactory.get(activation),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.decoupled_encoder = DecoupledEncoder(
            self.latent_dim, n_feature, hidden_dim, drop_prob, batch_norm, layer_norm, activation
        )
        decoder_hidden_dim = hidden_dim if isinstance(hidden_dim, (int, float)) else hidden_dim[-1]
        self.decoder = Decoder(
            self.latent_dim,
            n_feature,
            decoder_hidden_dim,
            drop_prob,
            batch_norm,
            layer_norm,
            activation='leaky_relu',
        )

        self.z_s_dom_dim = self.decoupled_encoder.z_s_dom_dim
        self.domain_classifier_s = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.z_s_dom_dim, self.n_domain),
        )
        self.solver = MirrorDescentNNLS(steps=20, init_eta=0.5, sum_to_one=False, learn_eta=True, eps=1e-8)

    def forward(self, x):
        z_c, z_s, z_s_dom_for_batch = self.decoupled_encoder.forward_with_batch_s(x)
        pi, p, z_hat = self.solve_pi(z_c)
        recon_x = self.decoder(z_c, z_s)
        domain_logits_s = self.domain_classify_s(z_s_dom_for_batch)
        return ForwardOut(
            z_c=z_c,
            z_s=z_s,
            pi=pi,
            p=p,
            z_hat=z_hat,
            recon_x=recon_x,
            domain_logits_s=domain_logits_s,
        )

    def solve_pi(self, z_c: torch.Tensor):
        z_c = z_c / (z_c.norm(dim=-1, keepdim=True) + 1e-8)
        S_z_proj = self.proto_projector(self.S_z_raw)
        S_z_proj = S_z_proj / (S_z_proj.norm(dim=-1, keepdim=True) + 1e-8)
        pi = self.solver(z_c, S_z_proj)
        z_hat = pi @ S_z_proj
        B = pi.size(0)
        C = int(self.num_celltypes)
        p = torch.zeros(B, C, device=pi.device, dtype=pi.dtype)
        p.scatter_add_(1, self.proto2ct.view(1, -1).expand(B, -1), pi)
        p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
        return pi, p, z_hat

    def domain_classify_s(self, z_s_or_dom):
        if z_s_or_dom.size(-1) == self.latent_dim:
            z_s_or_dom = z_s_or_dom[:, :self.z_s_dom_dim]
        return self.domain_classifier_s(z_s_or_dom)

    def _run_epoch_losses(
        self,
        dataloader,
        device,
        loss_weight,
        train_mode=True,
        optimizer=None,
        epoch: int = 0,
    ):
        track_keys = [
            "L_total",
            "L_prop",
            "L_rec",
            "L_latrec",
            "L_domain",
            "L_align",
            "L_contrast",
        ]
        history = {key: [] for key in track_keys}
        self.train(train_mode)
        context = torch.enable_grad() if train_mode else torch.no_grad()

        with context:
            for pseudo, real, _ in dataloader:
                x_pseudo = pseudo["x"].float().to(device)
                domain_pseudo = pseudo["domain"].long().to(device)
                prop_pseudo = pseudo["prop_celltype"].float().to(device)
                x_real = real["x"].float().to(device)
                domain_real = real["domain"].long().to(device)

                out_pseudo = self(x_pseudo)
                out_real = self(x_real)
                losses = compute_losses(
                    out_pseudo,
                    out_real,
                    x_pseudo,
                    x_real,
                    prop_pseudo,
                    domain_pseudo,
                    domain_real,
                    loss_weight,
                    epoch=epoch,
                )

                if train_mode and optimizer is not None:
                    optimizer.zero_grad()
                    losses["L_total"].backward()
                    optimizer.step()

                for key in track_keys:
                    history[key].append(losses[key].detach().item())

        return {
            key: float(np.mean(values)) if values else float("nan")
            for key, values in history.items()
        }

    @staticmethod
    def _format_losses(prefix: str, values: dict) -> str:
        return (
            f"{prefix}: total={values['L_total']:.4f} "
            f"prop={values['L_prop']:.4f} "
            f"rec={values['L_rec']:.4f} "
            f"latrec={values['L_latrec']:.4f} "
            f"domain={values['L_domain']:.4f} "
            f"align={values['L_align']:.4f} "
            f"contrast={values['L_contrast']:.4f}"
        )

    def fit(
        self,
        train_dataloader,
        val_dataloader=None,
        lr=1e-4,
        weight_decay=1e-3,
        max_epoch=500,
        device="cuda",
        patience=10,
        loss_weight=LossWeights(),
        outdir=None,
        verbose=False,
        logger=None,
    ):
        self.to(device)

        ckpt_path = outdir if outdir else os.path.join(os.getcwd(), "DECIPHER.pt")
        ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
        os.makedirs(ckpt_dir, exist_ok=True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

        trace_func = logger.info if logger is not None else print
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=verbose,
            path=ckpt_path,
            trace_func=trace_func,
        )

        track_keys = [
            "L_total",
            "L_prop",
            "L_rec",
            "L_latrec",
            "L_domain",
            "L_align",
            "L_contrast",
        ]
        history = {"train": {key: [] for key in track_keys}}
        if val_dataloader is not None:
            history["val"] = {key: [] for key in track_keys}

        for epoch in range(max_epoch):
            train_values = self._run_epoch_losses(
                train_dataloader,
                device,
                loss_weight,
                train_mode=True,
                optimizer=optimizer,
                epoch=epoch,
            )
            val_values = None
            if val_dataloader is not None:
                val_values = self._run_epoch_losses(
                    val_dataloader,
                    device,
                    loss_weight,
                    train_mode=False,
                    epoch=epoch,
                )
                monitor_loss = val_values["L_total"]
            else:
                monitor_loss = train_values["L_total"]

            for key in track_keys:
                history["train"][key].append(train_values[key])
                if val_values is not None:
                    history["val"][key].append(val_values[key])

            log_message = (
                f"[Epoch {epoch + 1:03d}/{max_epoch}] "
                f"{self._format_losses('train', train_values)}"
            )
            if val_values is not None:
                log_message += f" | {self._format_losses('val', val_values)}"
            if logger is not None:
                logger.info(log_message)
            elif verbose:
                print(log_message)

            early_stopping(monitor_loss, self)
            if early_stopping.early_stop:
                break

        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            cur = self.state_dict()
            compatible = {
                k: v for k, v in ckpt.items()
                if k in cur and cur[k].shape == v.shape
            }
            self.load_state_dict(compatible, strict=False)
        self.eval()
        self.loss_history_ = history
        return history

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
        self.latent_dim = int(latent_dim)
        self.z_s_dom_dim = self.latent_dim // 2
        self.z_s_free_dim = self.latent_dim - self.z_s_dom_dim

        self.common_encoder = MultiDenseLayer(n_feature, hidden_dim, drop_prob, 
                                        batch_norm, layer_norm, activation)
        last_hidden_dim = hidden_dim if isinstance(hidden_dim, (int, float)) else hidden_dim[-1]

        self.z_c_dense = nn.Sequential(
            nn.Linear(last_hidden_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim, elementwise_affine=False)
        )

        # Split z_s into two independent projection heads:
        #   z_s_dom  = first half, domain latent supervised by L_batch_s
        #   z_s_free = second half
        # This avoids LayerNorm coupling between the two halves.
        self.z_s_dom_dense = nn.Sequential(
            nn.Linear(last_hidden_dim, self.z_s_dom_dim),
            nn.LayerNorm(self.z_s_dom_dim, elementwise_affine=False)
        )
        self.z_s_free_dense = nn.Sequential(
            nn.Linear(last_hidden_dim, self.z_s_free_dim),
            nn.LayerNorm(self.z_s_free_dim, elementwise_affine=False)
        )

    def _encode_from_h(self, h):
        z_c = self.z_c_dense(h)
        z_s_dom = self.z_s_dom_dense(h)
        z_s_free = self.z_s_free_dense(h)
        z_s = torch.cat([z_s_dom, z_s_free], dim=-1)
        return z_c, z_s, z_s_dom

    def forward(self, x):
        """Return z_c and the full z_s used by the normal model path."""
        h = self.common_encoder(x)
        z_c, z_s, _ = self._encode_from_h(h)
        return z_c, z_s

    def forward_with_batch_s(self, x):
        """
        Return normal latents plus a special z_s_dom branch for L_batch_s.

        Batch_s path:
            h.detach() -> z_s_dom_dense -> z_s_dom_for_batch -> domain_classifier_s
            L_batch_s updates z_s_dom_dense and domain_classifier_s, but not common_encoder.
        """
        h = self.common_encoder(x)
        z_c, z_s, _ = self._encode_from_h(h)
        z_s_dom_for_batch = self.z_s_dom_dense(h.detach())
        # z_s_dom_for_batch = self.z_s_dom_dense(h)
        return z_c, z_s, z_s_dom_for_batch


class Decoder(nn.Module):
    def __init__(self, latent_dim, n_feature, hidden_dim, drop_prob,
                 batch_norm=False, layer_norm=True,
                 activation='leaky_relu'):
        super().__init__()
        self.dense1 = MultiDenseLayer(latent_dim * 2, hidden_dim, drop_prob,
                                     batch_norm, layer_norm, activation)
        last_hidden_dim = hidden_dim if isinstance(hidden_dim, (int, float)) else hidden_dim[-1]
        self.dense2 = nn.Sequential(
            nn.Linear(last_hidden_dim, n_feature),
            ActivationFactory.get('relu')
        )

    def forward(self, z_c, z_s):
        h = self.dense1(torch.cat([z_c, z_s], dim=-1))
        return self.dense2(h)

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