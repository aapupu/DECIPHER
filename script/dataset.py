import torch
from torch.utils.data import Dataset
from typing import Optional

from dataprocess import DecodePseudoBulkBuilder, DecodePseudoBulkConfig, _rng

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
