"""Gap 1 — cross-dataset leave-one-out loader (binary label only).

Train on the union of N-1 datasets, test on the held-out one. z-score is fit on
the union TRAIN only and applied to the held-out test set, so the standardiser
never sees target-domain statistics. Binary `Label` (B-tax-1) -> no class-code
alignment needed across datasets.
"""
from __future__ import annotations

import os
import json

import numpy as np

from src import data as D


def _load_raw(name, max_rows, parquet=None, per_class_cap=None, seed=42,
              lean=False):
    feat_cols, X, _attacks, y_bin = D._read_dataset(
        name, parquet or D.DEFAULT_PARQUET, max_rows,
        per_class_cap=per_class_cap, seed=seed)
    # lean keeps the float32 the reader produced. The standardisation below
    # then widens block by block, so the values match the float64 path.
    return (feat_cols, X if lean else X.astype(np.float64),
            y_bin.astype(np.int64))


def load_loo(train, test, parquet=None, max_rows=None, cache_dir=None,
             per_class_cap=None, seed=42, drop_test_dupes=True, lean=False):
    """Train on the union of `train` datasets, return held-out `test` set.

    Returns (Xtr, ytr, Xte, yte, meta). Binary label. z-score fit on union train.
    Pass `per_class_cap` (e.g. 400_000) to bound memory on the big datasets via a
    stratified per-class reservoir (see data._read_dataset). `seed` drives that
    reservoir's subsampling and is forwarded to every per-dataset read.

    `drop_test_dupes` (default True) removes held-out rows whose exact feature
    content also occurs in the training union. The four NF corpora overlap:
    measured on a 150,000-row prefix, NF-ToN-IoT-v2 shares 33.67 percent of its
    rows with the union of the other three, NF-BoT-IoT-v2 3.88 and
    NF-CSE-CIC-IDS2018-v2 0.73. Without this the leave-one-out result is partly
    an in-distribution result. Pass False only to reproduce the pre-2026-07-20
    numbers in results/gap1/. See audit defect B1.
    """
    train = list(train)
    # A held-out corpus that also appears in `train` would collapse to a single
    # key in the dict below and silently be used as both train and test.
    assert test not in train, (
        f"held-out dataset {test!r} also appears in the training union "
        f"{train!r}; leave-one-out requires the held-out corpus to be excluded "
        f"from train")
    raws = {n: _load_raw(n, max_rows, parquet, per_class_cap, seed=seed,
                         lean=lean)
            for n in (train + [test])}
    # shared feature columns across all datasets, stable order from first train ds
    common = [c for c in raws[train[0]][0]
              if all(c in raws[n][0] for n in raws)]

    def select(name):
        cols, X, y = raws[name]
        idx = [cols.index(c) for c in common]
        return X[:, idx], y

    Xtr = np.concatenate([select(n)[0] for n in train], axis=0)
    ytr = np.concatenate([select(n)[1] for n in train], axis=0)
    Xte, yte = select(test)
    # Free the per-dataset float64 raws before the hashing/z-score copies:
    # peak host memory otherwise exceeds a 14 GB box. Numerics unchanged.
    del raws, select

    # --- B1 cross-dataset leak check, on the shared columns, pre-z-score ---
    # Keys are taken before the constant-column drop and the z-score so "same
    # row" means the same measured flow, not the same standardised vector.
    n_test_raw = len(Xte)
    n_dupes = 0
    if drop_test_dupes:
        dup = np.isin(D.row_hashes_fast(Xte), D.row_hashes_fast(Xtr))
        n_dupes = int(dup.sum())
        Xte, yte = Xte[~dup], yte[~dup]

    if lean:
        # Memory-lean standardisation (14 GB box): float64 accumulators and
        # block-wise widening, never a full float64 copy of the union.
        n_tr, blk = len(Xtr), 500_000
        s1 = np.zeros(Xtr.shape[1])
        for i in range(0, n_tr, blk):
            s1 += Xtr[i:i + blk].sum(axis=0, dtype=np.float64)
        mu = s1 / n_tr
        s2 = np.zeros(Xtr.shape[1])
        for i in range(0, n_tr, blk):
            s2 += ((Xtr[i:i + blk].astype(np.float64) - mu) ** 2).sum(axis=0)
        sd = np.sqrt(s2 / n_tr)
        keep = sd > 0
        common = [c for c, k in zip(common, keep) if k]
        mu, sd = mu[keep], sd[keep]
        Xtr_out = np.empty((n_tr, int(keep.sum())), dtype=np.float32)
        for i in range(0, n_tr, blk):
            Xtr_out[i:i + blk] = (
                (Xtr[i:i + blk][:, keep].astype(np.float64) - mu) / sd)
        Xtr, Xte = Xtr_out, ((Xte[:, keep].astype(np.float64) - mu)
                             / sd).astype(np.float32)
    else:
        # drop constant cols on union TRAIN only
        keep = Xtr.std(axis=0) > 0
        common = [c for c, k in zip(common, keep) if k]
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        # z-score on union train
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        Xtr = ((Xtr - mu) / sd).astype(np.float32)
        Xte = ((Xte - mu) / sd).astype(np.float32)
    meta = {"train": list(train), "test": test, "n_features": len(common),
            "features": common, "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
            "dedup": bool(drop_test_dupes),
            "row_key": "fnv1a64",
            "row_key_collision_bound": float((n_test_raw + len(Xtr)) ** 2
                                             / 2.0 ** 65),
            "n_test_raw": int(n_test_raw),
            "n_test_dupes_dropped": int(n_dupes),
            "test_dupe_frac": float(n_dupes / n_test_raw) if n_test_raw else 0.0}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "loo_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
    return Xtr, ytr, Xte, yte, meta


def split_train_val_disjoint(X, val_frac=0.15, seed=42):
    """Carve validation indices whose row content does not also occur in train.

    Rows are assigned uniformly at random first, then any validation row whose
    exact content still occurs among the training rows is dropped from
    validation; its copies stay in train, so the training set never shrinks.
    Mirrors the leak check `data.materialize_dataset` already applies to its
    own three-way split.

    Returns (train_idx, val_idx, stats). Index arrays, not copies: the caller
    keeps its own view on X and no second multi-GB array is allocated.

    Degenerate case: a pool where every row is duplicated leaves validation
    empty, which would make val-only checkpoint selection impossible. One row is
    then kept and `stats["degenerate"]` is set so the caller can record it.
    """
    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(val_frac * n))
    va, tr = perm[:n_val], perm[n_val:]

    # Key the pool once and index the keys, never the rows. `X[va]` and `X[tr]`
    # are fancy indexes, so each would copy its slice: at 5.1M rows by 39 float32
    # that is about 800 MB for the train side alone, on top of the widening
    # temporary inside row_hashes_fast. The key array is 8 bytes per row, so the
    # whole pool costs about 41 MB instead.
    keys = D.row_hashes_fast(X)
    dup = np.isin(keys[va], keys[tr])
    n_dropped = int(dup.sum())
    va_clean = va[~dup]
    degenerate = len(va_clean) == 0
    if degenerate:
        va_clean = va[:1]

    stats = {"n_val_raw": int(n_val), "n_val_dupes_dropped": n_dropped,
             "n_val": int(len(va_clean)), "n_train": int(len(tr)),
             "val_dupe_frac": float(n_dropped / n_val) if n_val else 0.0,
             "degenerate": bool(degenerate)}
    return tr, va_clean, stats
