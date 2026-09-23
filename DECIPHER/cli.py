"""Command-line entry for DECIPHER deconvolution."""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataprocess import (
    DecodePseudoBulkBuilder,
    DecodePseudoBulkConfig,
    process_adata,
    reorder_celltype_proportions,
    row_max_normalize,
)
from .loss import LossWeights
from .mixup import PseudoRealDynamicPairDataset, pair_collate
from .model import DECIPHER
from .utils import seed_everything


def _load_bulk_matrix(path: Path, gene_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Load bulk as (n_sample, n_gene) aligned to ``gene_names``."""
    genes = [str(g) for g in gene_names]
    if path.suffix == ".h5ad":
        bulk = sc.read_h5ad(path)
        var = bulk.var_names.astype(str)
        missing = [g for g in genes if g not in var]
        if missing:
            raise KeyError(f"bulk .h5ad missing {len(missing)} genes, e.g. {missing[:5]}")
        X = bulk[:, genes].X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32), bulk.obs_names.astype(str).tolist()

    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    if set(genes).issubset(df.index):
        # genes × samples
        sub = df.loc[genes]
        return sub.to_numpy(dtype=np.float32).T, sub.columns.tolist()
    if set(genes).issubset(df.columns):
        # samples × genes
        sub = df.loc[:, genes]
        return sub.to_numpy(dtype=np.float32), sub.index.tolist()
    raise KeyError(
        "bulk CSV must have genes as index or columns matching the single-cell reference"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="DECIPHER",
        description="DECIPHER cellular deconvolution (train + predict)",
    )
    p.add_argument("--sc_path", required=True, help="Single-cell reference .h5ad")
    p.add_argument("--bulk_path", required=True, help="Bulk / mixture .csv or .h5ad")
    p.add_argument("--celltype_key", default="celltype", help="obs column for cell types")
    p.add_argument("--outdir", default="DECIPHER_out", help="Output directory")
    p.add_argument("--n_pseudo", type=int, default=4000)
    p.add_argument("--cells_per_bulk", type=int, default=200)
    p.add_argument("--n_pca", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_epoch", type=int, default=300)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--real_mixup_prob", type=float, default=0.0)
    p.add_argument("--real_mixup_k", type=int, default=4)
    p.add_argument("--real_mixup_alpha", type=float, default=0.5)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    seed_everything(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.sc_path)
    if args.celltype_key not in adata.obs.columns:
        raise KeyError(f"obs missing celltype key {args.celltype_key!r}")
    celltype_order = list(dict.fromkeys(adata.obs[args.celltype_key].astype(str)))
    gene_names = adata.var_names.astype(str).tolist()
    sc_data = process_adata(
        adata,
        celltype_key=args.celltype_key,
        state_key=None,
        gene_names=gene_names,
    )

    X_sc = np.asarray(sc_data.X, dtype=np.float32)
    n_comp = min(args.n_pca, X_sc.shape[0], X_sc.shape[1])
    Z = PCA(n_components=n_comp, random_state=args.seed).fit_transform(X_sc)
    labels = adata.obs[args.celltype_key].astype(str).to_numpy()
    s_z = torch.from_numpy(
        np.stack([Z[labels == ct].mean(0) for ct in celltype_order]).astype(np.float32)
    )

    builder = DecodePseudoBulkBuilder(
        sc_data,
        DecodePseudoBulkConfig(cells_per_bulk=args.cells_per_bulk, seed=args.seed),
        require_state=False,
    )
    pseudo_x, pseudo_prop = [], []
    for _ in range(args.n_pseudo):
        x, p, _ = builder.build_one()
        pseudo_x.append(x)
        pseudo_prop.append(
            reorder_celltype_proportions(p, sc_data.celltype_map, celltype_order)
        )
    pseudo_x = row_max_normalize(np.stack(pseudo_x))
    pseudo_prop = np.stack(pseudo_prop).astype(np.float32)

    genes_used = sc_data.gene_names or gene_names
    bulk_X, sample_ids = _load_bulk_matrix(Path(args.bulk_path), genes_used)
    bulk_X = row_max_normalize(bulk_X)
    real_domain = np.ones(bulk_X.shape[0], dtype=np.int64)

    idx_train, idx_val = train_test_split(
        np.arange(pseudo_x.shape[0]), test_size=0.2, random_state=args.seed
    )
    train_ds = PseudoRealDynamicPairDataset(
        pseudo_x[idx_train],
        pseudo_prop[idx_train],
        bulk_X,
        real_domain,
        seed=args.seed,
        real_mixup_prob=args.real_mixup_prob,
        real_mixup_k=args.real_mixup_k,
        real_mixup_alpha=args.real_mixup_alpha,
        real_sampling="random",
    )
    val_ds = PseudoRealDynamicPairDataset(
        pseudo_x[idx_val],
        pseudo_prop[idx_val],
        bulk_X,
        real_domain,
        seed=args.seed + 1,
        real_mixup_prob=0.0,
        real_sampling="fixed_cycle",
    )
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pair_collate,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pair_collate
    )

    model = DECIPHER(
        s_z=s_z,
        n_feature=pseudo_x.shape[1],
        hidden_dim=(128, 128),
        n_domain=2,
    ).to(device)
    ckpt = outdir / "DECIPHER.pt"
    model.fit(
        train_loader,
        val_dataloader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epoch=args.max_epoch,
        device=device,
        patience=args.patience,
        loss_weight=LossWeights(),
        outdir=str(ckpt),
        verbose=args.verbose,
    )

    _, proportions = model.deconvolution(torch.from_numpy(bulk_X).to(device))
    pred_df = pd.DataFrame(
        proportions.detach().cpu().numpy(),
        index=sample_ids,
        columns=celltype_order,
    )
    pred_path = outdir / "pred_proportions.csv"
    pred_df.to_csv(pred_path)
    print(f"saved checkpoint: {ckpt}")
    print(f"saved predictions: {pred_path}")


if __name__ == "__main__":
    main()
