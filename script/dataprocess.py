# data_module_decipher.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, List, Union

import numpy as np
import torch

try:
    import scipy.sparse as sp
except Exception:
    sp = None


# -----------------------------
# helpers
# -----------------------------
def _to_dense_float32(X) -> np.ndarray:
    """AnnData X can be dense or sparse; return dense float32 ndarray."""
    if sp is not None and sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X)
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    return X

def _normalize_cpm_log1p(counts: np.ndarray, eps: float = 1e-8, scale: float = 1e6) -> np.ndarray:
    """
    counts: [N, G] raw counts
    returns log1p(CPM)
    """
    lib = counts.sum(axis=1, keepdims=True) + eps
    x = counts / lib * scale
    return np.log1p(x).astype(np.float32)

def _subset_genes(X: np.ndarray, gene_idx: np.ndarray) -> np.ndarray:
    return X[:, gene_idx]

def _encode_labels(values: Sequence[Any]) -> Tuple[np.ndarray, Dict[Any, int]]:
    uniq = list(dict.fromkeys(values))  # stable
    mapping = {k: i for i, k in enumerate(uniq)}
    ids = np.array([mapping[v] for v in values], dtype=np.int64)
    return ids, mapping

def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)

# -----------------------------
# preprocessing
# -----------------------------
@dataclass
class SCProcessed:
    X: np.ndarray                  # [N_cells, G]
    celltype_id: np.ndarray        # [N_cells]
    state_id: Optional[np.ndarray] # [N_cells] if provided
    gene_names: Optional[List[str]]
    celltype_map: Dict[Any, int]
    state_map: Optional[Dict[Any, int]]


@dataclass
class BulkProcessed:
    X: np.ndarray            # [N_bulk, G]
    domain_id: np.ndarray    # [N_bulk] int
    batch_list: List[Any]
    batch_map: Dict[Any, int]


def process_adata(
    adata,
    celltype_key: str = "celltype",
    state_key: Optional[str] = "celltype_state",
    use_raw: bool = False,
    assume_log1p_cp10k: bool = False,
    hvg_mask: Optional[np.ndarray] = None,
    gene_names: Optional[List[str]] = None,
) -> SCProcessed:
    """
    adata: AnnData
    - If assume_log1p_cp10k=False, treat adata.X as raw counts and do CP10k+log1p.
    - If hvg_mask provided, subset genes.
    """
    X = adata.raw.X if use_raw and getattr(adata, "raw", None) is not None else adata.X
    X = _to_dense_float32(X)

    if hvg_mask is not None:
        idx = np.where(np.asarray(hvg_mask).astype(bool))[0]
        X = _subset_genes(X, idx)
        if gene_names is not None:
            gene_names = [gene_names[i] for i in idx]

    if not assume_log1p_cp10k:
        X = _normalize_cpm_log1p(X, scale=1e4)

    obs = adata.obs
    ct_vals = obs[celltype_key].astype(str).tolist()
    celltype_id, celltype_map = _encode_labels(ct_vals)

    state_id = None
    state_map = None
    if state_key is not None and state_key in obs.columns:
        st_vals = obs[state_key].astype(str).tolist()
        state_id, state_map = _encode_labels(st_vals)

    return SCProcessed(
        X=X,
        celltype_id=celltype_id,
        state_id=state_id,
        gene_names=gene_names,
        celltype_map=celltype_map,
        state_map=state_map,
    )


def process_bulk_dict(
    bulk_dict: Dict[str, Any],
    assume_log1p_cp10k: bool = False,
    hvg_mask: Optional[np.ndarray] = None,
) -> BulkProcessed:
    """
    bulk_dict:
      - 'batch_list': list-like length N_bulk
      - 'gex_matrix': [N_bulk, G] counts or already log1p
    """
    X = bulk_dict["gex_matrix"]
    X = np.asarray(X, dtype=np.float32)

    if hvg_mask is not None:
        idx = np.where(np.asarray(hvg_mask).astype(bool))[0]
        X = X[:, idx]

    if not assume_log1p_cp10k:
        X = _normalize_cpm_log1p(X, scale=1e6)

    batch_list = list(bulk_dict["batch_list"])
    # batch_id: 0..K-1
    batch_id, batch_map = _encode_labels([str(b) for b in batch_list])
    # domain_id for real bulk: 1..K
    domain_id = batch_id + 1

    return BulkProcessed(X=X, domain_id=domain_id, batch_list=batch_list, batch_map=batch_map)

@dataclass
class DecodePseudoBulkConfig:
    cells_per_bulk: int = 200
    seed: Optional[int] = 0
    uniform_low: float = 0.0
    uniform_high: float = 1.0

    # optional realism knobs
    add_library_size_jitter: bool = False
    lib_jitter_logsigma: float = 0.3
    add_gene_bias: bool = False
    gene_bias_logsigma: float = 0.05


class DecodePseudoBulkBuilder:
    """
    Scheme A:
      - sample celltype proportions with DECODE Stage-1 rule (Uniform -> normalize)
      - sample cells per celltype via round(p*m) + adjust
      - label celltype proportions = realized n_i/sum(n)
      - ALSO compute state proportions by counting sampled cells' state_id
    """
    def __init__(self, sc, cfg: DecodePseudoBulkConfig, require_state: bool = True):
        """
        sc must provide:
          sc.X: [N_cells, G]
          sc.celltype_id: [N_cells]
          sc.state_id: [N_cells] (optional if require_state=False)
        """
        self.sc = sc
        self.cfg = cfg
        self.rng = _rng(cfg.seed)

        self.ct = sc.celltype_id
        self.num_celltypes = int(self.ct.max()) + 1

        self.state = sc.state_id
        if require_state and self.state is None:
            raise ValueError("Scheme A requires sc.state_id (celltype_state).")
        self.num_states = int(self.state.max()) + 1 if self.state is not None else 0

        # indices by celltype
        self.ct_indices: List[np.ndarray] = []
        for c in range(self.num_celltypes):
            idx = np.where(self.ct == c)[0]
            if idx.size == 0:
                raise ValueError(f"celltype {c} has 0 cells.")
            self.ct_indices.append(idx)

        self.G = sc.X.shape[1]

    def _sample_p_celltype_decode(self) -> np.ndarray:
        r = self.rng.uniform(self.cfg.uniform_low, self.cfg.uniform_high, size=self.num_celltypes).astype(np.float32)
        s = float(r.sum())
        if s <= 0:
            r = np.ones((self.num_celltypes,), dtype=np.float32)
            s = float(r.sum())
        return (r / s).astype(np.float32)

    def _round_counts(self, p: np.ndarray, m: int) -> np.ndarray:
        n_i = np.rint(p * m).astype(int)
        n_i[n_i < 0] = 0
        if n_i.sum() == 0:
            n_i[self.rng.integers(0, self.num_celltypes)] = 1

        # adjust to exactly m (optional but recommended for consistency)
        total = int(n_i.sum())
        if total != m:
            diff = m - total
            for _ in range(abs(diff)):
                j = int(self.rng.integers(0, self.num_celltypes))
                if diff > 0:
                    n_i[j] += 1
                else:
                    if n_i[j] > 0:
                        n_i[j] -= 1
        return n_i

    def build_one(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        returns:
          x_pseudo: [G] float32
          p_celltype: [C] float32  (realized proportions after rounding)
          p_state:    [S] float32  (from sampled cells; None if state_id not provided)
        """
        m = int(self.cfg.cells_per_bulk)

        # 1-2) DECODE: uniform -> normalize
        p = self._sample_p_celltype_decode()         # [C]

        # 3) n_i = round(p*m) and adjust
        n_i = self._round_counts(p, m)               # [C]

        # 4) update label by realized counts
        p_celltype = (n_i / max(n_i.sum(), 1)).astype(np.float32)

        # 5) aggregate sampled cells; meanwhile collect sampled indices for state counting
        x_sum = np.zeros((self.G,), dtype=np.float32)
        sampled_state_counts = None
        if self.state is not None:
            sampled_state_counts = np.zeros((self.num_states,), dtype=np.int64)

        for c, n in enumerate(n_i):
            if n <= 0:
                continue
            pool = self.ct_indices[c]
            pick = self.rng.choice(pool, size=n, replace=(n > pool.size))
            x_sum += self.sc.X[pick].sum(axis=0)

            if sampled_state_counts is not None:
                st = self.state[pick]
                # fast bincount
                sampled_state_counts += np.bincount(st, minlength=self.num_states)

        # optional realism knobs
        if self.cfg.add_library_size_jitter:
            scale = float(np.exp(self.rng.normal(0.0, self.cfg.lib_jitter_logsigma)))
            x_sum *= scale
        if self.cfg.add_gene_bias:
            bias = np.exp(self.rng.normal(0.0, self.cfg.gene_bias_logsigma, size=self.G)).astype(np.float32)
            x_sum *= bias

        # state proportions
        p_state = None
        if sampled_state_counts is not None:
            p_state = (sampled_state_counts / max(sampled_state_counts.sum(), 1)).astype(np.float32)

        return x_sum, p_celltype, p_state
