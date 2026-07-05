from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn.functional as F


def gated_latrec_loss(
    z_c: torch.Tensor,
    z_hat: torch.Tensor,
    pi: torch.Tensor,
    alpha: float = 5.0,
    beta: float = 10.0,
    tau_entropy: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    error = ((z_c - z_hat) ** 2).mean(dim=-1)
    entropy = -(pi * (pi + eps).log()).sum(dim=-1)
    error_gate = torch.exp(-alpha * error)
    entropy_gate = torch.sigmoid(beta * (tau_entropy - entropy))
    weight = (error_gate * entropy_gate).detach()
    return (weight * error).mean()


def CORAL(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.size(0) < 2 or target.size(0) < 2:
        return source.new_tensor(0.0)

    feature_dim = source.size(1)
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)
    source_cov = source_centered.T @ source_centered / (source.size(0) - 1)
    target_cov = target_centered.T @ target_centered / (target.size(0) - 1)
    return torch.linalg.norm(source_cov - target_cov, ord="fro").pow(2) / (
        4 * feature_dim**2
    )


def mmd2_rbf(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_mul: float = 1.5,
    kernel_num: int = 3,
    fix_sigma2: Optional[float] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    n_x, n_y = x.size(0), y.size(0)
    if n_x == 0 or n_y == 0:
        return x.new_tensor(0.0)

    values = torch.cat([x, y], dim=0)
    distance_sq = torch.cdist(values, values, p=2).pow(2)

    if fix_sigma2 is None:
        with torch.no_grad():
            upper_mask = torch.triu(
                torch.ones_like(distance_sq, dtype=torch.bool),
                diagonal=1,
            )
            base_sigma2 = (0.5 * distance_sq[upper_mask].median()).clamp_min(eps)
    else:
        base_sigma2 = x.new_tensor(fix_sigma2).clamp_min(eps)

    kernel = torch.zeros_like(distance_sq)
    for index in range(kernel_num):
        sigma2 = base_sigma2 * (kernel_mul ** (index - kernel_num // 2))
        kernel += torch.exp(-distance_sq / (2.0 * sigma2 + eps))
    kernel /= kernel_num

    kernel_xx = kernel[:n_x, :n_x]
    kernel_yy = kernel[n_x:, n_x:]
    kernel_xy = kernel[:n_x, n_x:]
    return (kernel_xx.mean() + kernel_yy.mean() - 2.0 * kernel_xy.mean()).clamp_min(0.0)


def cmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    n_moments: int = 3,
) -> torch.Tensor:
    if source.size(0) < 2 or target.size(0) < 2:
        return source.new_tensor(0.0)

    loss = torch.norm(source.mean(dim=0) - target.mean(dim=0), p=2)
    source_centered = source - source.mean(dim=0, keepdim=True)
    target_centered = target - target.mean(dim=0, keepdim=True)

    for moment in range(2, n_moments + 1):
        source_moment = (source_centered**moment).mean(dim=0)
        target_moment = (target_centered**moment).mean(dim=0)
        loss += torch.norm(source_moment - target_moment, p=2) / moment

    return loss / n_moments


def proportion_contrastive_loss(
    z_c: torch.Tensor,
    proportions: torch.Tensor,
    temperature: float = 0.2,
    pos_threshold: float = 0.9,
    eps: float = 1e-8,
) -> torch.Tensor:
    batch_size = z_c.size(0)
    if batch_size < 2:
        return z_c.new_tensor(0.0)

    z_c = F.normalize(z_c, dim=1)
    proportions = F.normalize(proportions, dim=1)
    latent_similarity = z_c @ z_c.T / temperature
    proportion_similarity = proportions @ proportions.T

    self_mask = torch.eye(batch_size, device=z_c.device, dtype=torch.bool)
    positive_mask = (proportion_similarity > pos_threshold) & ~self_mask
    if positive_mask.sum() == 0:
        return z_c.new_tensor(0.0)

    logits = latent_similarity.masked_fill(self_mask, -1e9)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive_mask.float().sum(dim=1)
    loss = -(log_probability * positive_mask.float()).sum(dim=1) / (
        positive_count + eps
    )
    return loss[positive_count > 0].mean()


@dataclass
class LossWeights:
    w_prop: float = 1.0
    w_rec: float = 0.2
    w_latrec: float = 1.0
    w_dom: float = 0.2
    w_align: float = 1.0
    w_contrast: float = 0.1
    mse_ce_ratio: float = 4.0
    ema_decay: float = 0.99
    align_auto: bool = True
    align_probe_epochs: int = 2
    align_open_thresh: float = 0.5
    _ema_mse: float = field(default=0.0, repr=False)
    _ema_ce: float = field(default=0.0, repr=False)
    _ema_init: bool = field(default=False, repr=False)
    _ema_d_within: float = field(default=0.0, repr=False)
    _ema_d_between: float = field(default=0.0, repr=False)
    _ema_align_init: bool = field(default=False, repr=False)
    _align_enabled: bool = field(default=False, repr=False)
    _align_frozen: bool = field(default=False, repr=False)


def _proportion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weights: LossWeights,
) -> torch.Tensor:
    mse = F.mse_loss(prediction, target)
    cross_entropy = -(target * (prediction + 1e-8).log()).sum(dim=-1).mean()

    with torch.no_grad():
        mse_value = float(mse.detach().abs().item())
        ce_value = float(cross_entropy.detach().abs().item()) + 1e-8
        if not weights._ema_init:
            weights._ema_mse = mse_value
            weights._ema_ce = ce_value
            weights._ema_init = True
        else:
            weights._ema_mse = (
                weights.ema_decay * weights._ema_mse
                + (1.0 - weights.ema_decay) * mse_value
            )
            weights._ema_ce = (
                weights.ema_decay * weights._ema_ce
                + (1.0 - weights.ema_decay) * ce_value
            )
        ce_weight = weights._ema_mse / (
            weights.mse_ce_ratio * weights._ema_ce + 1e-8
        )

    return mse + ce_weight * cross_entropy


def _alignment_weight(
    z_real: torch.Tensor,
    alignment_loss: torch.Tensor,
    weights: LossWeights,
    epoch: int,
) -> float:
    if not weights.align_auto:
        return float(weights.w_align)

    with torch.no_grad():
        batch_size = z_real.size(0)
        if batch_size >= 4:
            permutation = torch.randperm(batch_size, device=z_real.device)
            midpoint = batch_size // 2
            within_loss = mmd2_rbf(
                z_real[permutation[:midpoint]],
                z_real[permutation[midpoint : 2 * midpoint]],
            )
        else:
            within_loss = z_real.new_tensor(0.0)

        between_value = float(alignment_loss.detach().abs().item())
        within_value = float(within_loss.detach().abs().item())
        if not weights._ema_align_init:
            weights._ema_d_between = between_value
            weights._ema_d_within = within_value
            weights._ema_align_init = True
        else:
            weights._ema_d_between = (
                weights.ema_decay * weights._ema_d_between
                + (1.0 - weights.ema_decay) * between_value
            )
            weights._ema_d_within = (
                weights.ema_decay * weights._ema_d_within
                + (1.0 - weights.ema_decay) * within_value
            )

        if epoch < weights.align_probe_epochs:
            return 0.0
        if epoch == weights.align_probe_epochs and not weights._align_frozen:
            weights._align_enabled = (
                weights._ema_d_between < weights.align_open_thresh
            )
            weights._align_frozen = True
            return 0.0
        if not weights._align_frozen:
            weights._align_enabled = (
                weights._ema_d_between < weights.align_open_thresh
            )
            weights._align_frozen = True

    return float(weights.w_align) if weights._align_enabled else 0.0


def compute_losses(
    out_pseudo,
    out_real,
    x_pseudo: torch.Tensor,
    x_real: torch.Tensor,
    p_true: torch.Tensor,
    domain_pseudo: torch.Tensor,
    domain_real: torch.Tensor,
    weights: LossWeights,
    epoch: int = 0,
):
    losses = {}
    losses["L_prop"] = _proportion_loss(out_pseudo.p, p_true, weights)
    losses["L_rec"] = 0.5 * (
        F.mse_loss(out_pseudo.recon_x, x_pseudo)
        + F.mse_loss(out_real.recon_x, x_real)
    )
    losses["L_latrec"] = 0.5 * (
        gated_latrec_loss(out_pseudo.z_c, out_pseudo.z_hat, out_pseudo.pi)
        + gated_latrec_loss(out_real.z_c, out_real.z_hat, out_real.pi)
    )

    domain_labels = torch.cat([domain_pseudo, domain_real], dim=0).long()
    domain_logits = torch.cat(
        [out_pseudo.domain_logits, out_real.domain_logits],
        dim=0,
    )
    losses["L_domain"] = F.cross_entropy(domain_logits, domain_labels)

    losses["L_align"] = mmd2_rbf(out_pseudo.z_c.detach(), out_real.z_c)
    alignment_weight = _alignment_weight(
        out_real.z_c.detach(),
        losses["L_align"],
        weights,
        epoch,
    )
    losses["L_contrast"] = proportion_contrastive_loss(out_pseudo.z_c, p_true)

    losses["L_total"] = (
        weights.w_prop * losses["L_prop"]
        + weights.w_rec * losses["L_rec"]
        + weights.w_latrec * losses["L_latrec"]
        + weights.w_dom * losses["L_domain"]
        + alignment_weight * losses["L_align"]
        + weights.w_contrast * losses["L_contrast"]
    )
    return losses
