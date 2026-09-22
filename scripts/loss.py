from dataclasses import dataclass, field
import torch
import torch.nn.functional as F


def gated_latrec_loss(
    zc: torch.Tensor,
    zhat: torch.Tensor,
    pi: torch.Tensor,
    alpha: float = 5.0,
    beta: float = 10.0,
    tau_entropy: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    err = ((zc - zhat) ** 2).mean(dim=-1)
    ent = -(pi * (pi + eps).log()).sum(dim=-1)
    w_err = torch.exp(-alpha * err)
    w_ent = torch.sigmoid(beta * (tau_entropy - ent))
    w = (w_err * w_ent).detach()
    return (w * err).mean()


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
    fix_sigma2: float = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    n, m = x.size(0), y.size(0)
    if n == 0 or m == 0:
        return x.new_tensor(0.0)

    z = torch.cat([x, y], dim=0)
    dist2 = torch.cdist(z, z, p=2).pow(2)

    if fix_sigma2 is not None:
        base_sigma2 = x.new_tensor(fix_sigma2).clamp_min(eps)
    else:
        with torch.no_grad():
            tri = torch.triu(torch.ones_like(dist2, dtype=torch.bool), diagonal=1)
            med = dist2[tri].median()
            base_sigma2 = (0.5 * med).clamp_min(eps)

    sigma2_list = [
        base_sigma2 * (kernel_mul ** (i - kernel_num // 2))
        for i in range(kernel_num)
    ]

    kernel = torch.zeros_like(dist2)
    for sigma2 in sigma2_list:
        kernel = kernel + torch.exp(-dist2 / (2.0 * sigma2 + eps))
    kernel = kernel / kernel_num

    kernel_xx = kernel[:n, :n]
    kernel_yy = kernel[n:, n:]
    kernel_xy = kernel[:n, n:]
    return (kernel_xx.mean() + kernel_yy.mean() - 2.0 * kernel_xy.mean()).clamp_min(0.0)


def proportion_contrastive_loss(
    z: torch.Tensor,
    p: torch.Tensor,
    temperature: float = 0.2,
    pos_threshold: float = 0.9,
    eps: float = 1e-8,
) -> torch.Tensor:
    batch_size = z.size(0)
    if batch_size < 2:
        return z.new_tensor(0.0)

    z = F.normalize(z, dim=1)
    p = F.normalize(p, dim=1)
    z_sim = torch.matmul(z, z.T) / temperature
    p_sim = torch.matmul(p, p.T)

    self_mask = torch.eye(batch_size, device=z.device).bool()
    pos_mask = (p_sim > pos_threshold) & (~self_mask)
    if pos_mask.sum() == 0:
        return z.new_tensor(0.0)

    logits = z_sim.masked_fill(self_mask, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    loss = -(log_prob * pos_mask.float()).sum(dim=1) / (pos_mask.float().sum(dim=1) + eps)
    valid = pos_mask.float().sum(dim=1) > 0
    return loss[valid].mean()


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


def compute_losses(
    out_pseudo,
    out_real,
    x_pseudo: torch.Tensor,
    x_real: torch.Tensor,
    p_true: torch.Tensor,
    domain_pseudo: torch.Tensor,
    domain_real: torch.Tensor,
    w: LossWeights,
    epoch: int = 0,
):
    losses = {}

    eps = 1e-8
    mse = F.mse_loss(out_pseudo.p, p_true)
    ce = -(p_true * (out_pseudo.p + eps).log()).sum(dim=-1).mean()

    with torch.no_grad():
        m_abs = float(mse.detach().abs().item())
        c_abs = float(ce.detach().abs().item()) + 1e-8
        if not w._ema_init:
            w._ema_mse, w._ema_ce, w._ema_init = m_abs, c_abs, True
        else:
            w._ema_mse = w.ema_decay * w._ema_mse + (1.0 - w.ema_decay) * m_abs
            w._ema_ce = w.ema_decay * w._ema_ce + (1.0 - w.ema_decay) * c_abs
        lam = w._ema_mse / (w.mse_ce_ratio * w._ema_ce + 1e-8)

    losses["L_prop"] = mse + lam * ce

    losses["L_rec"] = 0.5 * (
        F.mse_loss(out_pseudo.recon_x, x_pseudo)
        + F.mse_loss(out_real.recon_x, x_real)
    )

    losses["L_latrec"] = 0.5 * (
        gated_latrec_loss(out_pseudo.z_c, out_pseudo.z_hat, out_pseudo.pi)
        + gated_latrec_loss(out_real.z_c, out_real.z_hat, out_real.pi)
    )

    domain_labels = torch.cat([domain_pseudo, domain_real], dim=0).long()
    dom_logits_s = torch.cat(
        [out_pseudo.domain_logits_s, out_real.domain_logits_s],
        dim=0,
    )
    losses["L_domain"] = F.cross_entropy(dom_logits_s, domain_labels)

    d_between = mmd2_rbf(out_pseudo.z_c.detach(), out_real.z_c)
    losses["L_align"] = d_between

    with torch.no_grad():
        zr = out_real.z_c.detach()
        batch_size = zr.size(0)
        if batch_size >= 4:
            perm = torch.randperm(batch_size, device=zr.device)
            half = batch_size // 2
            d_within = mmd2_rbf(zr[perm[:half]], zr[perm[half:2 * half]])
        else:
            d_within = zr.new_tensor(0.0)

        db = float(d_between.detach().abs().item())
        dw = float(d_within.detach().abs().item())
        if not w._ema_align_init:
            w._ema_d_between, w._ema_d_within, w._ema_align_init = db, dw, True
        else:
            w._ema_d_between = w.ema_decay * w._ema_d_between + (1.0 - w.ema_decay) * db
            w._ema_d_within = w.ema_decay * w._ema_d_within + (1.0 - w.ema_decay) * dw

        if w.align_auto:
            if epoch < w.align_probe_epochs:
                w_eff = 0.0
            elif epoch == w.align_probe_epochs:
                if not w._align_frozen:
                    w._align_enabled = w._ema_d_between < w.align_open_thresh
                    w._align_frozen = True
                w_eff = 0.0
            else:
                if not w._align_frozen:
                    w._align_enabled = w._ema_d_between < w.align_open_thresh
                    w._align_frozen = True
                w_eff = float(w.w_align) if w._align_enabled else 0.0
        else:
            w_eff = float(w.w_align)

    losses["L_contrast"] = proportion_contrastive_loss(
        out_pseudo.z_c,
        p_true,
        temperature=0.2,
        pos_threshold=0.9,
    )

    losses["L_total"] = (
        w.w_prop * losses["L_prop"]
        + w.w_rec * losses["L_rec"]
        + w.w_latrec * losses["L_latrec"]
        + w.w_dom * losses["L_domain"]
        + w_eff * losses["L_align"]
        + w.w_contrast * losses["L_contrast"]
    )
    return losses

