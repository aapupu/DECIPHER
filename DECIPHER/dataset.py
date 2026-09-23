from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from .dataprocess import DecodePseudoBulkBuilder, DecodePseudoBulkConfig, _rng


class PseudoRealPairDataset(Dataset):
    """On-the-fly pseudo synthesis paired with real bulk samples."""

    def __init__(
        self,
        sc,
        bulk,
        pseudo_cfg: DecodePseudoBulkConfig,
        length: int = 20_000,
        real_sampling: str = "random",
        seed: Optional[int] = 0,
        pseudo_domain_id: int = 0,
        require_state: bool = False,
    ):
        self.bulk = bulk
        self.builder = DecodePseudoBulkBuilder(
            sc,
            pseudo_cfg,
            require_state=require_state,
        )
        self.length = int(length)
        self.real_sampling = real_sampling
        self.rng = _rng(seed)
        self.n_real = bulk.X.shape[0]
        self.real_pointer = 0
        self.pseudo_domain_id = int(pseudo_domain_id)

        if bulk.X.shape[1] != sc.X.shape[1]:
            raise ValueError("bulk and sc must have the same gene dimension.")

    def __len__(self) -> int:
        return self.length

    def _sample_real_index(self) -> int:
        if self.real_sampling == "random":
            return int(self.rng.integers(0, self.n_real))
        if self.real_sampling == "cycle":
            index = self.real_pointer
            self.real_pointer = (self.real_pointer + 1) % self.n_real
            return index
        raise ValueError("real_sampling must be 'random' or 'cycle'.")

    def __getitem__(self, index: int):
        x_pseudo, prop_celltype, prop_state = self.builder.build_one()
        if prop_state is None:
            prop_state = np.zeros(
                max(1, self.builder.num_states),
                dtype=np.float32,
            )

        real_index = self._sample_real_index()
        pseudo = {
            "x": torch.from_numpy(x_pseudo),
            "prop_celltype": torch.from_numpy(prop_celltype),
            "prop_state": torch.from_numpy(prop_state),
            "domain": torch.tensor(self.pseudo_domain_id, dtype=torch.long),
        }
        real = {
            "x": torch.from_numpy(self.bulk.X[real_index]),
            "domain": torch.tensor(
                int(self.bulk.domain_id[real_index]),
                dtype=torch.long,
            ),
        }
        return pseudo, real, {}


class PseudoRealDynamicPairDataset(Dataset):
    """Precomputed pseudo arrays; random real bulk per index (no mixup)."""

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
        real_sampling: str = "random",
    ):
        self.X_pseudo = np.asarray(X_pseudo, dtype=np.float32)
        self.props = np.asarray(props_pseudo, dtype=np.float32)
        self.bulk_X = np.asarray(bulk_X, dtype=np.float32)
        self.bulk_domain_id = np.asarray(bulk_domain_id, dtype=np.int64)
        self.pseudo_domain_id = int(pseudo_domain_id)
        self.n_real = int(self.bulk_X.shape[0])
        self.rng = _rng(seed)
        self.real_sampling = str(real_sampling)
        if self.real_sampling not in {"random", "fixed_cycle"}:
            raise ValueError("real_sampling must be 'random' or 'fixed_cycle'")

        if self.X_pseudo.shape[0] != self.props.shape[0]:
            raise ValueError("X_pseudo and props_pseudo must have the same length.")
        if self.X_pseudo.shape[1] != self.bulk_X.shape[1]:
            raise ValueError("pseudo and real bulk must have the same gene dimension.")

        if props_state_pseudo is not None:
            self.props_state = np.asarray(props_state_pseudo, dtype=np.float32)
            if self.props_state.shape[0] != self.X_pseudo.shape[0]:
                raise ValueError("props_state_pseudo must match X_pseudo length.")
        else:
            self.props_state = None
            self.num_state_props = int(max(1, num_state_props))

    def __len__(self) -> int:
        return self.X_pseudo.shape[0]

    def __getitem__(self, index: int):
        if self.real_sampling == "fixed_cycle":
            real_index = int(index) % self.n_real
        else:
            real_index = int(self.rng.integers(0, self.n_real))
        prop_state = (
            torch.from_numpy(self.props_state[index])
            if self.props_state is not None
            else torch.zeros(self.num_state_props, dtype=torch.float32)
        )
        pseudo = {
            "x": torch.from_numpy(self.X_pseudo[index]),
            "prop_celltype": torch.from_numpy(self.props[index]),
            "prop_state": prop_state,
            "domain": torch.tensor(self.pseudo_domain_id, dtype=torch.long),
        }
        real = {
            "x": torch.from_numpy(self.bulk_X[real_index]),
            "domain": torch.tensor(
                int(self.bulk_domain_id[real_index]),
                dtype=torch.long,
            ),
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
