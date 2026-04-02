import torch
import torch.nn as nn
import torch.nn.functional as F 

from dataclasses import dataclass

def cross_covariance_penalty(zc: torch.Tensor, zs: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """||Cov(zc, zs)||_F^2 on standardized features."""
    if zc.size(0) < 2:
        return zc.new_tensor(0.0)
    zc = (zc - zc.mean(0, keepdim=True)) / (zc.std(0, keepdim=True) + eps)
    zs = (zs - zs.mean(0, keepdim=True)) / (zs.std(0, keepdim=True) + eps)
    cov = (zc.t() @ zs) / (zc.size(0) - 1)
    return (cov ** 2).mean()

def cosine_loss(z_c, z_s):
    cos_sim = F.cosine_similarity(z_c, z_s, dim=1)
    return torch.mean(cos_sim**2)

def gated_latrec_loss(
    zc: torch.Tensor,
    zhat: torch.Tensor,
    pi: torch.Tensor,
    alpha: float = 5.0,
    beta: float = 10.0,
    tau_entropy: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Return mean( w_i * ||zc_i - zhat_i||^2 ), w_i in (0,1).
    """
    # per-sample recon error
    err = ((zc - zhat) ** 2).mean(dim=-1)  # [B]

    # entropy (higher -> more uniform -> less confident)
    ent = -(pi * (pi + eps).log()).sum(dim=-1)  # [B]

    w_err = torch.exp(-alpha * err)                 # [B]
    w_ent = torch.sigmoid(beta * (tau_entropy - ent))  # [B]
    w = (w_err * w_ent).detach()                    # IMPORTANT: detach gate

    return (w * err).mean()

def CORAL(source, target):
    d = source.size(1)

    # center the features
    xm = source - torch.mean(source, dim=0, keepdim=True)
    xmt = target - torch.mean(target, dim=0, keepdim=True)

    # covariance matrices with unbiased estimator (optional)
    Cs = (xm.t() @ xm) / (source.size(0) - 1)
    Ct = (xmt.t() @ xmt) / (target.size(0) - 1)

    # frobenius norm between covariance matrices
    squared_fro_norm = torch.linalg.norm(Cs - Ct, ord="fro") ** 2
    return squared_fro_norm / (4 * (d**2))

@dataclass
class LossWeights:
    w_prop: float = 1.0
    w_rec: float = 0.2
    w_latrec: float = 1.0
    w_dom: float = 0.2
    w_decouple: float = 0.1
    w_align: float = 1

def compute_losses(
    out_pseudo,
    out_real,
    x_pseudo: torch.Tensor,
    x_real: torch.Tensor,
    p_true: torch.Tensor,          # [B_pseudo, C]
    domain_pseudo: torch.Tensor,   # [B_pseudo] int (0..n_domain-1)
    domain_real: torch.Tensor,     # [B_real] int
    w: LossWeights,
):
    """
    - proportion supervision on pseudo only
    - reconstruction on both
    - latent reconstruction on both (can set w_latrec=0 for ablation)
    - domain classification on both (zs subspace)
    - decoupling zc ⟂ zs (optional)
    """
    losses = {}

    # (1) proportion supervision (pseudo only)
    eps = 1e-8
    ce = -(p_true * (out_pseudo.p + eps).log()).sum(dim=-1).mean()
    losses["L_prop"] = (out_pseudo.p - p_true).abs().mean() + 0.1*ce

    # (2) reconstruction loss (both)
    losses["L_rec"] = 0.5 * (F.mse_loss(out_pseudo.recon_x, x_pseudo) + 
                             F.mse_loss(out_real.recon_x, x_real))

    # (3) latent reconstruction (both)
    losses["L_latrec"] = 0.5 * (gated_latrec_loss(out_pseudo.z_c, out_pseudo.z_hat) + 
                                gated_latrec_loss(out_real.z_c, out_real.z_hat))

    # (4) domain prediction loss (both)
    dom_logits = torch.cat([out_pseudo.domain_logits, out_real.domain_logits], dim=0)  # [B, n_domain]
    dom_labels = torch.cat([domain_pseudo, domain_real], dim=0).long()
    losses["L_dom"] = F.cross_entropy(dom_logits, dom_labels)

    # (5) decoupling (optional)
    zc_all = torch.cat([out_pseudo.z_c, out_real.z_c], dim=0)
    zs_all = torch.cat([out_pseudo.z_s, out_real.z_s], dim=0)
    losses["L_decouple"] = cosine_loss(zc_all, zs_all)


    # (8) alignment (optional)
    losses["L_align"] = CORAL(out_pseudo.z_c.detach(), out_real.z_c)

    losses["L_total"] = (
        w.w_prop * losses["L_prop"]
        + w.w_rec * losses["L_rec"]
        + w.w_latrec * losses["L_latrec"]
        + w.w_dom * losses["L_dom"]
        + w.w_decouple * losses["L_decouple"]
        +  w.w_align * losses["L_align"]
    )
    return losses
