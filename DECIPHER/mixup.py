"""Real-mixture mixup for DECIPHER training.

During training, with probability ``real_mixup_prob``, sample ``k`` real
mixtures and combine them with Dirichlet(``alpha``) weights. Default
settings used in the paper:

  real_mixup_prob=0.2, real_mixup_k=4, real_mixup_alpha=0.5, mode="dirichlet"

Validation should use ``real_sampling="fixed_cycle"`` and ``real_mixup_prob=0``.
"""

from __future__ import annotations
from typing import Optional, Sequence
import numpy as np
import torch
from torch.utils.data import Dataset
from .dataprocess import _rng


class PseudoRealDynamicPairDataset(Dataset):
    """Precomputed pseudo arrays paired with (optionally mixed) real bulks."""

    def __init__(
        self,
        X_pseudo: np.ndarray,
        props_pseudo: np.ndarray,
        bulk_X: np.ndarray,
        bulk_domain_id: np.ndarray,
        seed: int,
        pseudo_domain_id: int = 0,
        num_state_props: int = 1,
        props_state_pseudo: Optional[np.ndarray] = None,
        bulk_ct_mask: Optional[np.ndarray] = None,
        real_mixup_prob: float = 0.0,
        real_mixup_k: int = 4,
        real_mixup_alpha: float = 0.5,
        real_mixup_mode: str = "dirichlet",
        real_mixup_k_choices: Optional[Sequence[int]] = None,
        real_mixup_k_probs: Optional[Sequence[float]] = None,
        real_mixup_group_id: Optional[np.ndarray] = None,
        real_sampling: str = "random",
    ):
        """
        real_sampling:
          - "random": each access samples a real (optional mixup). For training.
          - "fixed_cycle": real_index = index % n_real, no mixup. Deterministic val.
        real_mixup_mode (only when mixup fires under random sampling):
          - "dirichlet": weights ~ Dirichlet(alpha, ..., alpha)
          - "uniform": equal weights 1/k
        real_mixup_group_id:
          - optional per-real group labels; mixup stays within one group.
        """
        self.X_pseudo = np.asarray(X_pseudo, dtype=np.float32)
        self.props = np.asarray(props_pseudo, dtype=np.float32)
        self.bulk_X = np.asarray(bulk_X, dtype=np.float32)
        self.bulk_domain_id = np.asarray(bulk_domain_id, dtype=np.int64)
        self.pseudo_domain_id = int(pseudo_domain_id)
        self.n_real = int(self.bulk_X.shape[0])
        self.rng = _rng(seed)
        self.real_mixup_prob = float(real_mixup_prob)
        self.real_mixup_k = int(real_mixup_k)
        self.real_mixup_alpha = float(real_mixup_alpha)
        self.real_mixup_mode = str(real_mixup_mode)
        self.real_sampling = str(real_sampling)
        if self.real_sampling not in {"random", "fixed_cycle"}:
            raise ValueError("real_sampling must be 'random' or 'fixed_cycle'")
        if self.real_mixup_mode not in {"dirichlet", "uniform"}:
            raise ValueError("real_mixup_mode must be 'dirichlet' or 'uniform'")

        if real_mixup_k_choices is not None:
            choices = np.asarray(list(real_mixup_k_choices), dtype=np.int64)
            if choices.ndim != 1 or choices.size < 1:
                raise ValueError("real_mixup_k_choices must be a non-empty 1-d sequence")
            if np.any(choices < 2):
                raise ValueError("all real_mixup_k_choices must be >= 2")
            self.real_mixup_k_choices = choices
            if real_mixup_k_probs is None:
                probs = np.full(choices.size, 1.0 / choices.size, dtype=np.float64)
            else:
                probs = np.asarray(list(real_mixup_k_probs), dtype=np.float64)
                if probs.shape != choices.shape:
                    raise ValueError("real_mixup_k_probs must match real_mixup_k_choices")
                if np.any(probs < 0) or float(probs.sum()) <= 0:
                    raise ValueError("real_mixup_k_probs must be non-negative and sum > 0")
                probs = probs / probs.sum()
            self.real_mixup_k_probs = probs
            self._mixup_k_max = int(choices.max())
        else:
            self.real_mixup_k_choices = None
            self.real_mixup_k_probs = None
            self._mixup_k_max = int(self.real_mixup_k)

        if self.X_pseudo.shape[0] != self.props.shape[0]:
            raise ValueError("X_pseudo and props_pseudo must have the same length.")
        if self.X_pseudo.shape[1] != self.bulk_X.shape[1]:
            raise ValueError("pseudo and real bulk must have the same gene dimension.")

        if bulk_ct_mask is not None:
            self.bulk_ct_mask = np.asarray(bulk_ct_mask, dtype=np.float32)
            if self.bulk_ct_mask.shape[0] != self.n_real:
                raise ValueError(
                    f"bulk_ct_mask rows ({self.bulk_ct_mask.shape[0]}) "
                    f"must match bulk_X rows ({self.n_real})"
                )
            self.n_ct = int(self.bulk_ct_mask.shape[1])
        else:
            self.bulk_ct_mask = None
            self.n_ct = int(self.props.shape[1]) if self.props.ndim == 2 else 1

        if real_mixup_group_id is not None:
            self.real_mixup_group_id = np.asarray(real_mixup_group_id, dtype=np.int64)
            if self.real_mixup_group_id.shape[0] != self.n_real:
                raise ValueError(
                    f"real_mixup_group_id length ({self.real_mixup_group_id.shape[0]}) "
                    f"must match bulk_X rows ({self.n_real})"
                )
            self._mixup_group_ids = np.unique(self.real_mixup_group_id)
            self._mixup_group_indices = {
                int(g): np.flatnonzero(self.real_mixup_group_id == g)
                for g in self._mixup_group_ids
            }
            if self.real_mixup_prob > 0.0 and self.real_sampling == "random":
                too_small = [
                    int(g)
                    for g in self._mixup_group_ids
                    if self._mixup_group_indices[int(g)].size < self._mixup_k_max
                ]
                if too_small:
                    raise ValueError(
                        f"mixup groups too small for k={self._mixup_k_max}: {too_small}"
                    )
        else:
            self.real_mixup_group_id = None
            self._mixup_group_ids = None
            self._mixup_group_indices = None

        if self.real_mixup_prob > 0.0 and self.real_sampling == "random":
            if self._mixup_k_max < 2:
                raise ValueError("mixup k must be >= 2 when mixup is enabled")
            if self.real_mixup_group_id is None and self.n_real < self._mixup_k_max:
                raise ValueError(
                    f"n_real={self.n_real} < max mixup k={self._mixup_k_max}"
                )

        if props_state_pseudo is not None:
            self.props_state = np.asarray(props_state_pseudo, dtype=np.float32)
            if self.props_state.shape[0] != self.X_pseudo.shape[0]:
                raise ValueError("props_state_pseudo must match X_pseudo length.")
        else:
            self.props_state = None
            self.num_state_props = int(max(1, num_state_props))

    def __len__(self) -> int:
        return self.X_pseudo.shape[0]

    def _get_real_at(self, real_index: int):
        x = self.bulk_X[real_index]
        domain = int(self.bulk_domain_id[real_index])
        if self.bulk_ct_mask is not None:
            ct_mask = self.bulk_ct_mask[real_index]
        else:
            ct_mask = np.ones(self.n_ct, dtype=np.float32)
        return x, domain, ct_mask

    def _sample_real(self, index: int):
        if self.real_sampling == "fixed_cycle":
            return self._get_real_at(int(index) % self.n_real)

        do_mixup = (
            self.real_mixup_prob > 0.0
            and float(self.rng.random()) < self.real_mixup_prob
        )
        if do_mixup:
            if self.real_mixup_k_choices is not None:
                k = int(
                    self.rng.choice(
                        self.real_mixup_k_choices, p=self.real_mixup_k_probs
                    )
                )
            else:
                k = int(self.real_mixup_k)
            if self._mixup_group_indices is not None:
                group = int(self.rng.choice(self._mixup_group_ids))
                pool = self._mixup_group_indices[group]
                idxs = self.rng.choice(pool, size=k, replace=False)
            else:
                idxs = self.rng.choice(self.n_real, size=k, replace=False)
            if self.real_mixup_mode == "uniform":
                w = np.full(k, 1.0 / k, dtype=np.float32)
            else:
                w = self.rng.dirichlet(
                    np.full(k, self.real_mixup_alpha, dtype=np.float64)
                ).astype(np.float32)
            x = (w[:, None] * self.bulk_X[idxs]).sum(axis=0).astype(np.float32)
            domain = int(self.bulk_domain_id[idxs[0]])
            if self.bulk_ct_mask is not None:
                ct_mask = self.bulk_ct_mask[idxs].max(axis=0).astype(np.float32)
            else:
                ct_mask = np.ones(self.n_ct, dtype=np.float32)
            return x, domain, ct_mask

        return self._get_real_at(int(self.rng.integers(0, self.n_real)))

    def __getitem__(self, index: int):
        prop_state = (
            torch.from_numpy(self.props_state[index])
            if self.props_state is not None
            else torch.zeros(self.num_state_props, dtype=torch.float32)
        )
        x_real, domain_real, ct_mask = self._sample_real(index)
        pseudo = {
            "x": torch.from_numpy(self.X_pseudo[index]),
            "prop_celltype": torch.from_numpy(self.props[index]),
            "prop_state": prop_state,
            "domain": torch.tensor(self.pseudo_domain_id, dtype=torch.long),
        }
        real = {
            "x": torch.from_numpy(np.asarray(x_real, dtype=np.float32)),
            "domain": torch.tensor(domain_real, dtype=torch.long),
            "ct_mask": torch.from_numpy(np.asarray(ct_mask, dtype=np.float32)),
        }
        return pseudo, real, {}


def pair_collate(batch):
    """Collate that preserves optional ``ct_mask`` on the real branch."""
    pseudos, reals, _ = zip(*batch)
    pseudo = {
        "x": torch.stack([item["x"] for item in pseudos]),
        "prop_celltype": torch.stack([item["prop_celltype"] for item in pseudos]),
        "prop_state": torch.stack([item["prop_state"] for item in pseudos]),
        "domain": torch.stack([item["domain"] for item in pseudos]).view(-1),
    }
    real = {
        "x": torch.stack([item["x"] for item in reals]),
        "domain": torch.stack([item["domain"] for item in reals]).view(-1),
    }
    if "ct_mask" in reals[0]:
        real["ct_mask"] = torch.stack([item["ct_mask"] for item in reals])
    return pseudo, real, {}
