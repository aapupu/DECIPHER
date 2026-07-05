import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

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
    domain_logits: torch.Tensor


class DECIPHER(nn.Module):
    def __init__(
        self,
        s_z: torch.Tensor,
        n_feature: int = 2000,
        hidden_dim=(128, 128),
        proto2ct=None,
        n_celltypes=None,
        n_domain: int = 2,
        drop_prob: float = 0.1,
        batch_norm: bool = False,
        layer_norm: bool = True,
        activation: str = "leaky_relu",
        align_dim: int = 32,
    ):
        super().__init__()

        # Prepare prototypes
        if s_z.dim() == 3:
            n_celltypes_input, n_states, prototype_dim = s_z.shape
            prototypes = s_z.reshape(n_celltypes_input * n_states, prototype_dim)
            if proto2ct is None:
                proto2ct = torch.arange(n_celltypes_input).repeat_interleave(n_states)
            else:
                proto2ct = torch.as_tensor(proto2ct).long()
                if proto2ct.numel() != n_celltypes_input * n_states:
                    raise ValueError("proto2ct length must match the prototype count.")
            self.num_celltypes = (
                int(proto2ct.max().item()) + 1
                if n_celltypes is None
                else int(n_celltypes)
            )
            self.num_prototypes = n_celltypes_input * n_states
            self.n_cell_state = n_states
        elif s_z.dim() == 2:
            n_prototypes, prototype_dim = s_z.shape
            prototypes = s_z
            if proto2ct is None:
                proto2ct = torch.arange(n_prototypes)
                self.num_celltypes = n_prototypes
            else:
                proto2ct = torch.as_tensor(proto2ct).long()
                if proto2ct.numel() != n_prototypes:
                    raise ValueError("proto2ct length must match the prototype count.")
                self.num_celltypes = (
                    int(proto2ct.max().item()) + 1
                    if n_celltypes is None
                    else int(n_celltypes)
                )
            self.num_prototypes = n_prototypes
            self.n_cell_state = 0
        else:
            raise ValueError("s_z must be a 2D or 3D tensor.")

        prototypes = prototypes / (prototypes.norm(dim=-1, keepdim=True) + 1e-8)
        self.register_buffer("S_z_raw", prototypes)
        self.register_buffer("proto2ct", proto2ct.long())

        self.latent_dim = int(align_dim)
        self.n_domain = int(n_domain)
        self.proto_projector = nn.Sequential(
            nn.Linear(prototype_dim, self.latent_dim),
            ActivationFactory.get(activation),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.decoupled_encoder = DecoupledEncoder(
            self.latent_dim,
            n_feature,
            hidden_dim,
            drop_prob,
            batch_norm,
            layer_norm,
            activation,
        )
        decoder_hidden_dim = (
            hidden_dim if isinstance(hidden_dim, (int, float)) else hidden_dim[-1]
        )
        self.decoder = Decoder(
            self.latent_dim,
            n_feature,
            decoder_hidden_dim,
            drop_prob,
            batch_norm,
            layer_norm,
            activation="leaky_relu",
        )
        self.domain_classifier = nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Linear(self.decoupled_encoder.z_s_dom_dim, self.n_domain),
        )
        self.solver = MirrorDescentNNLS(
            steps=20,
            init_eta=0.5,
            sum_to_one=False,
            learn_eta=True,
            eps=1e-8,
        )

    def forward(self, x: torch.Tensor) -> ForwardOut:
        z_c, z_s, z_s_domain = self.decoupled_encoder.forward_with_domain_s(x)
        pi, p, z_hat = self.solve_pi(z_c)
        recon_x = self.decoder(z_c, z_s)
        domain_logits = self.domain_classifier(z_s_domain)
        return ForwardOut(
            z_c=z_c,
            z_s=z_s,
            pi=pi,
            p=p,
            z_hat=z_hat,
            recon_x=recon_x,
            domain_logits=domain_logits,
        )

    def solve_pi(self, z_c: torch.Tensor):
        z_c = z_c / (z_c.norm(dim=-1, keepdim=True) + 1e-8)
        projected_prototypes = self.proto_projector(self.S_z_raw)
        projected_prototypes = projected_prototypes / (
            projected_prototypes.norm(dim=-1, keepdim=True) + 1e-8
        )
        pi = self.solver(z_c, projected_prototypes)
        z_hat = pi @ projected_prototypes
        proportions = torch.zeros(
            pi.size(0),
            self.num_celltypes,
            device=pi.device,
            dtype=pi.dtype,
        )
        proportions.scatter_add_(
            1,
            self.proto2ct.view(1, -1).expand(pi.size(0), -1),
            pi,
        )
        proportions = proportions / (proportions.sum(dim=-1, keepdim=True) + 1e-8)
        return pi, proportions, z_hat

    def _run_epoch_losses(
        self,
        dataloader,
        device,
        loss_weights: LossWeights,
        train_mode: bool = True,
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
                p_true = pseudo["prop_celltype"].float().to(device)
                domain_pseudo = pseudo["domain"].long().to(device)
                x_real = real["x"].float().to(device)
                domain_real = real["domain"].long().to(device)

                out_pseudo = self(x_pseudo)
                out_real = self(x_real)
                losses = compute_losses(
                    out_pseudo,
                    out_real,
                    x_pseudo,
                    x_real,
                    p_true,
                    domain_pseudo,
                    domain_real,
                    loss_weights,
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
        lr: float = 1e-4,
        weight_decay: float = 1e-3,
        max_epoch: int = 500,
        device: str = "cuda",
        patience: int = 10,
        loss_weight: Optional[LossWeights] = None,
        outdir: Optional[str] = None,
        verbose: bool = False,
        logger=None,
    ):
        if loss_weight is None:
            loss_weight = LossWeights()

        self.to(device)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        checkpoint_path = outdir or os.path.join(os.getcwd(), "DECIPHER.pt")
        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        os.makedirs(checkpoint_dir, exist_ok=True)
        trace_func = logger.info if logger is not None else print
        early_stopping = EarlyStopping(
            patience=patience,
            verbose=False,
            path=checkpoint_path,
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
                monitored_loss = val_values["L_total"]
            else:
                monitored_loss = train_values["L_total"]

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

            early_stopping(monitored_loss, self)
            if early_stopping.early_stop:
                break

        if os.path.isfile(checkpoint_path):
            self.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.eval()
        self.loss_history_ = history
        return history

    @torch.no_grad()
    def encode_z(self, x: torch.Tensor):
        self.eval()
        return self.decoupled_encoder(x)

    @torch.no_grad()
    def deconvolution(self, x: torch.Tensor):
        self.eval()
        z_c, _ = self.decoupled_encoder(x)
        pi, proportions, _ = self.solve_pi(z_c)
        return pi, proportions


class MirrorDescentNNLS(nn.Module):
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

    def forward(self, z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        n_prototypes = prototypes.size(0)
        pi = torch.full(
            (batch_size, n_prototypes),
            1.0 / n_prototypes,
            device=z.device,
            dtype=z.dtype,
        )
        eta = self.eta.clamp(min=1e-4, max=10.0)

        for _ in range(self.steps):
            z_hat = pi @ prototypes
            gradient = 2.0 * (z_hat - z) @ prototypes.T
            pi = pi * torch.exp(-eta * gradient).clamp(min=self.eps)
            if self.sum_to_one:
                pi = pi / (pi.sum(dim=-1, keepdim=True) + self.eps)

        return pi


class DecoupledEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_feature: int,
        hidden_dim,
        drop_prob: float,
        batch_norm: bool,
        layer_norm: bool,
        activation: str,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.z_s_dom_dim = self.latent_dim // 2
        self.z_s_free_dim = self.latent_dim - self.z_s_dom_dim
        self.common_encoder = MultiDenseLayer(
            n_feature,
            hidden_dim,
            drop_prob,
            batch_norm,
            layer_norm,
            activation,
        )
        last_hidden_dim = (
            hidden_dim if isinstance(hidden_dim, (int, float)) else hidden_dim[-1]
        )
        self.z_c_dense = nn.Sequential(
            nn.Linear(last_hidden_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim, elementwise_affine=False),
        )
        self.z_s_dom_dense = nn.Sequential(
            nn.Linear(last_hidden_dim, self.z_s_dom_dim),
            nn.LayerNorm(self.z_s_dom_dim, elementwise_affine=False),
        )
        self.z_s_free_dense = nn.Sequential(
            nn.Linear(last_hidden_dim, self.z_s_free_dim),
            nn.LayerNorm(self.z_s_free_dim, elementwise_affine=False),
        )

    def _encode_from_hidden(self, hidden: torch.Tensor):
        z_c = self.z_c_dense(hidden)
        z_s_domain = self.z_s_dom_dense(hidden)
        z_s_free = self.z_s_free_dense(hidden)
        z_s = torch.cat([z_s_domain, z_s_free], dim=-1)
        return z_c, z_s, z_s_domain

    def forward(self, x: torch.Tensor):
        hidden = self.common_encoder(x)
        z_c, z_s, _ = self._encode_from_hidden(hidden)
        return z_c, z_s

    def forward_with_domain_s(self, x: torch.Tensor):
        hidden = self.common_encoder(x)
        z_c, z_s, _ = self._encode_from_hidden(hidden)
        z_s_domain = self.z_s_dom_dense(hidden.detach())
        return z_c, z_s, z_s_domain


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_feature: int,
        hidden_dim,
        drop_prob: float,
        batch_norm: bool = False,
        layer_norm: bool = True,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        self.dense1 = MultiDenseLayer(
            latent_dim * 2,
            hidden_dim,
            drop_prob,
            batch_norm,
            layer_norm,
            activation,
        )
        last_hidden_dim = (
            hidden_dim if isinstance(hidden_dim, (int, float)) else hidden_dim[-1]
        )
        self.dense2 = nn.Sequential(
            nn.Linear(last_hidden_dim, n_feature),
            ActivationFactory.get("relu"),
        )

    def forward(self, z_c: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        hidden = self.dense1(torch.cat([z_c, z_s], dim=-1))
        return self.dense2(hidden)


class DenseLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        drop_prob: float = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False,
        activation: str = "relu",
    ):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class MultiDenseLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden,
        drop_prob: float,
        batch_norm: bool = True,
        layer_norm: bool = False,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        if isinstance(hidden, (int, float)):
            hidden_dims = [int(hidden)]
        elif isinstance(hidden, (list, tuple)) and len(hidden) > 0:
            hidden_dims = [int(value) for value in hidden]
        else:
            raise ValueError("hidden must be a non-empty int, list, or tuple.")

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(
                DenseLayer(
                    current_dim,
                    hidden_dim,
                    drop_prob,
                    batch_norm=batch_norm,
                    layer_norm=layer_norm,
                    activation=activation,
                )
            )
            current_dim = hidden_dim
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ActivationFactory:
    _activations = {
        "relu": lambda: nn.ReLU(inplace=True),
        "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True),
        "gelu": lambda: nn.GELU(),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
        "softmax": lambda: nn.Softmax(-1),
        "none": lambda: nn.Identity(),
    }

    @staticmethod
    def get(name: str) -> nn.Module:
        if name not in ActivationFactory._activations:
            raise ValueError(f"Unsupported activation: {name}")
        return ActivationFactory._activations[name]()
