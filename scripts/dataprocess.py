from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy

try:
    import scanpy as sc
except ImportError:
    sc = None
from scipy.sparse import csr_matrix

try:
    import scipy.sparse as sp
except Exception:
    sp = None


def _to_dense_float32(X) -> np.ndarray:
    if sp is not None and sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _normalize_cpm_log1p(
    counts: np.ndarray,
    eps: float = 1e-8,
    scale: float = 1e6,
) -> np.ndarray:
    library_size = counts.sum(axis=1, keepdims=True) + eps
    return np.log1p(counts / library_size * scale).astype(np.float32)


def _subset_genes(X: np.ndarray, gene_idx: np.ndarray) -> np.ndarray:
    return X[:, gene_idx]


def _encode_labels(values: Sequence[Any]) -> Tuple[np.ndarray, Dict[Any, int]]:
    unique_values = list(dict.fromkeys(values))
    mapping = {value: index for index, value in enumerate(unique_values)}
    ids = np.array([mapping[value] for value in values], dtype=np.int64)
    return ids, mapping


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def row_max_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        return (X / (float(X.max()) + eps)).astype(np.float32)
    row_max = X.max(axis=1, keepdims=True)
    return (X / (row_max + eps)).astype(np.float32)




def reorder_celltype_proportions(
    p_ct: np.ndarray,
    celltype_map: Dict[Any, int],
    celltype_order: Sequence[Any],
) -> np.ndarray:
    """
    Align p_ct to a fixed cell-type order.

    p_ct[c] is the proportion for encoded id c (same layout as DecodePseudoBulkBuilder /
    process_adata celltype_id). celltype_map maps label -> id; celltype_order lists labels
    in the desired output order.
    """
    p_ct = np.asarray(p_ct, dtype=np.float32).reshape(-1)
    out = np.zeros((len(celltype_order),), dtype=np.float32)
    for i, name in enumerate(celltype_order):
        if name not in celltype_map:
            raise KeyError(f"celltype_order label {name!r} not in celltype_map")
        cid = int(celltype_map[name])
        if cid < 0 or cid >= p_ct.size:
            raise IndexError(
                f"celltype id {cid} for {name!r} out of range for p_ct size {p_ct.size}"
            )
        out[i] = p_ct[cid]
    return out


@dataclass
class SCProcessed:
    X: np.ndarray
    celltype_id: np.ndarray
    state_id: Optional[np.ndarray]
    gene_names: Optional[List[str]]
    celltype_map: Dict[Any, int]
    state_map: Optional[Dict[Any, int]]


@dataclass
class BulkProcessed:
    X: np.ndarray
    domain_id: np.ndarray
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
    X = adata.raw.X if use_raw and getattr(adata, "raw", None) is not None else adata.X
    X = _to_dense_float32(X)

    if hvg_mask is not None:
        gene_idx = np.where(np.asarray(hvg_mask).astype(bool))[0]
        X = _subset_genes(X, gene_idx)
        if gene_names is not None:
            gene_names = [gene_names[index] for index in gene_idx]

    if not assume_log1p_cp10k:
        X = _normalize_cpm_log1p(X, scale=1e4)

    obs = adata.obs
    celltype_values = obs[celltype_key].astype(str).tolist()
    celltype_id, celltype_map = _encode_labels(celltype_values)

    state_id = None
    state_map = None
    if state_key is not None and state_key in obs.columns:
        state_values = obs[state_key].astype(str).tolist()
        state_id, state_map = _encode_labels(state_values)

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
    assume_log1p_cp10k: bool = True,
    hvg_mask: Optional[np.ndarray] = None,
) -> BulkProcessed:
    X = np.asarray(bulk_dict["gex_matrix"], dtype=np.float32)

    if hvg_mask is not None:
        gene_idx = np.where(np.asarray(hvg_mask).astype(bool))[0]
        X = X[:, gene_idx]

    if not assume_log1p_cp10k:
        X = _normalize_cpm_log1p(X, scale=1e6)

    batch_list = list(bulk_dict["batch_list"])
    batch_id, batch_map = _encode_labels([str(batch) for batch in batch_list])
    domain_id = batch_id + 1

    return BulkProcessed(
        X=X,
        domain_id=domain_id,
        batch_list=batch_list,
        batch_map=batch_map,
    )


@dataclass
class DecodePseudoBulkConfig:
    cells_per_bulk: int = 200
    seed: Optional[int] = 0
    uniform_low: float = 0.0
    uniform_high: float = 1.0
    sampling_strategy: str = "uniform"
    min_nonzero: int = 1
    max_nonzero: int = 7
    lambda_nonzero: float = 3.0
    dirichlet_alpha: float = 1.0


class DecodePseudoBulkBuilder:
    def __init__(
        self,
        sc: SCProcessed,
        cfg: DecodePseudoBulkConfig,
        require_state: bool = True,
    ):
        self.sc = sc
        self.cfg = cfg
        self.rng = _rng(cfg.seed)
        self.celltype_id = sc.celltype_id
        self.num_celltypes = int(self.celltype_id.max()) + 1
        self.state_id = sc.state_id

        if require_state and self.state_id is None:
            raise ValueError("state_id is required when require_state=True.")

        self.num_states = (
            int(self.state_id.max()) + 1 if self.state_id is not None else 0
        )
        self.celltype_indices: List[np.ndarray] = []
        for celltype in range(self.num_celltypes):
            indices = np.where(self.celltype_id == celltype)[0]
            if indices.size == 0:
                raise ValueError(f"celltype {celltype} has no cells.")
            self.celltype_indices.append(indices)

        self.n_genes = sc.X.shape[1]

    def _sample_uniform_proportions(self) -> np.ndarray:
        proportions = self.rng.uniform(
            self.cfg.uniform_low,
            self.cfg.uniform_high,
            size=self.num_celltypes,
        ).astype(np.float32)
        total = float(proportions.sum())
        if total <= 0:
            proportions = np.ones(self.num_celltypes, dtype=np.float32)
            total = float(proportions.sum())
        return (proportions / total).astype(np.float32)

    def _sample_sparse_proportions(self) -> np.ndarray:
        lower = max(1, int(self.cfg.min_nonzero))
        upper = min(int(self.cfg.max_nonzero), self.num_celltypes)
        if lower > upper:
            raise ValueError("min_nonzero must not exceed max_nonzero.")

        for _ in range(10_000):
            n_selected = int(self.rng.poisson(float(self.cfg.lambda_nonzero)))
            if lower <= n_selected <= upper:
                break
        else:
            n_selected = int(np.clip(round(self.cfg.lambda_nonzero), lower, upper))

        selected = self.rng.choice(
            self.num_celltypes,
            size=n_selected,
            replace=False,
        )
        weights = self.rng.dirichlet(
            np.full(n_selected, float(self.cfg.dirichlet_alpha), dtype=np.float64)
        ).astype(np.float32)
        proportions = np.zeros(self.num_celltypes, dtype=np.float32)
        proportions[selected] = weights
        return proportions

    def _sample_celltype_proportions(self) -> np.ndarray:
        if self.cfg.sampling_strategy == "uniform":
            return self._sample_uniform_proportions()
        if self.cfg.sampling_strategy == "spatial_sparse":
            return self._sample_sparse_proportions()
        raise ValueError(f"Unknown sampling_strategy: {self.cfg.sampling_strategy!r}")

    def _round_counts(self, proportions: np.ndarray, total_cells: int) -> np.ndarray:
        counts = np.rint(proportions * total_cells).astype(int)
        counts[counts < 0] = 0

        if counts.sum() == 0:
            eligible = np.where(proportions > 0)[0]
            choice = int(self.rng.choice(eligible)) if eligible.size else 0
            counts[choice] = 1

        difference = total_cells - int(counts.sum())
        for _ in range(abs(difference)):
            if difference > 0:
                eligible = (
                    np.where(proportions > 0)[0]
                    if self.cfg.sampling_strategy == "spatial_sparse"
                    else np.arange(self.num_celltypes)
                )
                choice = int(self.rng.choice(eligible))
                counts[choice] += 1
            else:
                eligible = np.where(counts > 0)[0]
                choice = int(self.rng.choice(eligible))
                counts[choice] -= 1

        return counts

    def build_one(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        total_cells = int(self.cfg.cells_per_bulk)
        proportions = self._sample_celltype_proportions()
        cell_counts = self._round_counts(proportions, total_cells)
        celltype_proportions = (
            cell_counts / max(cell_counts.sum(), 1)
        ).astype(np.float32)

        pseudo_bulk = np.zeros(self.n_genes, dtype=np.float32)
        state_counts = (
            np.zeros(self.num_states, dtype=np.int64)
            if self.state_id is not None
            else None
        )

        for celltype, count in enumerate(cell_counts):
            if count <= 0:
                continue
            pool = self.celltype_indices[celltype]
            selected = self.rng.choice(pool, size=count, replace=count > pool.size)
            pseudo_bulk += self.sc.X[selected].sum(axis=0)
            if state_counts is not None:
                state_counts += np.bincount(
                    self.state_id[selected],
                    minlength=self.num_states,
                )

        state_proportions = None
        if state_counts is not None:
            state_proportions = (
                state_counts / max(state_counts.sum(), 1)
            ).astype(np.float32)

        return pseudo_bulk, celltype_proportions, state_proportions
