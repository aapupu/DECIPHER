from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from .dataprocess import DecodePseudoBulkBuilder, DecodePseudoBulkConfig, _rng


# On-the-fly pseudo synthesis each step; fixed virtual length; requires cell state.
class PseudoRealPairDataset(Dataset):
    """
    Each __getitem__ returns:
      pseudo: {'x': [G], 'prop_celltype': [C], 'prop_state': [S], 'domain': 0}
      real:   {'x': [G], 'domain': 1..K}
      meta:   {}
    """
    def __init__(
        self,
        sc,
        bulk,
        pseudo_cfg: DecodePseudoBulkConfig,
        length: int = 20000,
        real_sampling: str = "random",  # "random" or "cycle"
        seed: Optional[int] = 0,
        pseudo_domain_id: int = 0,
    ):
        self.sc = sc
        self.bulk = bulk
        self.builder = DecodePseudoBulkBuilder(sc, pseudo_cfg, require_state=True)

        self.length = int(length)
        self.real_sampling = real_sampling
        self.rng = _rng(seed)
        self.N_real = bulk.X.shape[0]
        self._real_ptr = 0

        self.G = sc.X.shape[1]
        assert bulk.X.shape[1] == self.G, "bulk and sc must have same gene dim after processing."

        self.C = self.builder.num_celltypes
        self.S = self.builder.num_states
        self.pseudo_domain_id = int(pseudo_domain_id)

    def __len__(self):
        return self.length

    def _sample_real_index(self):
        if self.real_sampling == "random":
            return int(self.rng.integers(0, self.N_real))
        idx = self._real_ptr
        self._real_ptr = (self._real_ptr + 1) % self.N_real
        return idx

    def __getitem__(self, i: int):
        x_p, p_ct, p_st = self.builder.build_one()

        ridx = self._sample_real_index()
        x_r = self.bulk.X[ridx]
        dom_r = int(self.bulk.domain_id[ridx])  # real domain should already be 1..K

        pseudo = {
            "x": torch.from_numpy(x_p),
            "prop_celltype": torch.from_numpy(p_ct),
            "prop_state": torch.from_numpy(p_st),
            "domain": torch.tensor(self.pseudo_domain_id, dtype=torch.long),
        }
        real = {
            "x": torch.from_numpy(x_r),
            "domain": torch.tensor(dom_r, dtype=torch.long),
        }
        meta = {}
        return pseudo, real, meta




# Precomputed pseudo arrays; random real bulk per index; default for decipher
class PseudoRealDynamicPairDataset(Dataset):
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
    ):
        self.X_pseudo = np.asarray(X_pseudo, dtype=np.float32)
        self.props = np.asarray(props_pseudo, dtype=np.float32)
        self.bulk_X = np.asarray(bulk_X, dtype=np.float32)
        self.bulk_domain_id = np.asarray(bulk_domain_id, dtype=np.int64)
        self.pseudo_domain_id = int(pseudo_domain_id)
        self.n_real = int(self.bulk_X.shape[0])
        self.rng = _rng(seed)

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
    pseudos, reals, metas = zip(*batch)
    pseudo = {
        "x": torch.stack([b["x"] for b in pseudos], dim=0),
        "prop_celltype": torch.stack([b["prop_celltype"] for b in pseudos], dim=0),
        "prop_state": torch.stack([b["prop_state"] for b in pseudos], dim=0),
        "domain": torch.stack([b["domain"] for b in pseudos], dim=0).view(-1),
    }
    real = {
        "x": torch.stack([b["x"] for b in reals], dim=0),
        "domain": torch.stack([b["domain"] for b in reals], dim=0).view(-1),
    }
    meta = {}
    return pseudo, real, meta
