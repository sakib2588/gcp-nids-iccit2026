"""T1 — NF-UQ-NIDS-v2 per-dataset loader.

Materialises clean, leakage-controlled, standardised per-dataset splits once to
disk, then loads them cheaply. Design decisions (plan v3 Part B):

- Drop the 4 flow IDENTIFIERS (IPs + ports) -> 39 numeric NF features remain.
- Drop constant columns (fit on TRAIN only).
- Stratified 70/15/15 draw on the NATIVE multiclass attack label, THEN a
  leakage-only de-duplication that deletes from val and test every row whose
  feature-hash occurs in train (train itself is left intact). The persisted
  ratios are therefore not 70/15/15; see `n_train`/`n_val`/`n_test` in meta.json.
- z-score standardiser fit on TRAIN only, applied to val/test.
- Persist X (float32), y_binary, y_multiclass per split + meta.json + checksums,
  so every pooling head consumes byte-identical input (T5e identical-input gate).
"""
from __future__ import annotations

import os
import json
import hashlib
import warnings
import collections

import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

# Path to the combined NF-UQ-NIDS-v2 parquet (download link in README.md).
# Override with the NF_UQ_PARQUET environment variable.
DEFAULT_PARQUET = os.environ.get(
    "NF_UQ_PARQUET",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "NF-UQ-NIDS-v2.parquet"),
)
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "data", "processed")

IDENTIFIER_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT"]
META_COLS = ["Label", "Attack", "Dataset"]
DATASETS = ["NF-UNSW-NB15-v2", "NF-CSE-CIC-IDS2018-v2",
            "NF-ToN-IoT-v2", "NF-BoT-IoT-v2"]


CLIP_ABS = 1e6   # clamp extreme NetFlow values before float32 (sibling streaming_loader)
NAN_FILL = 0.0   # raw-space value substituted for NaN (pre-standardisation)

# Sanitisation counters for the most recent `_read_dataset` call. Every substitution
# made by `_sanitise_features` is silent by construction, so it is counted here,
# raised as a warning at the end of the read, and recorded in meta.json under
# "sanitisation" by `materialize_dataset`. Zero counts are the expected case.
NONFINITE_COUNTS: dict = {"nan": 0, "posinf": 0, "neginf": 0, "clipped": 0}


def _reset_nonfinite_counts() -> None:
    for k in NONFINITE_COUNTS:
        NONFINITE_COUNTS[k] = 0


def _sanitise_features(feat: np.ndarray) -> np.ndarray:
    """Replace non-finite values, clamp to +/-CLIP_ABS, downcast to float32.

    Order matters. `np.clip` maps +inf and -inf onto the bounds but propagates NaN
    unchanged, so clipping alone does NOT keep NaN out of the network: a single NaN
    feature would reach the covariance pool and turn the whole batch NaN. NaN is
    therefore replaced explicitly by `np.nan_to_num`, with `posinf`/`neginf` pinned
    to the same bounds the clip enforces so the two steps cannot disagree.

    Substitution is imputation and imputation must never be silent, so counts land
    in NONFINITE_COUNTS and travel into meta.json.
    """
    if not np.isfinite(feat).all():
        NONFINITE_COUNTS["nan"] += int(np.isnan(feat).sum())
        NONFINITE_COUNTS["posinf"] += int(np.isposinf(feat).sum())
        NONFINITE_COUNTS["neginf"] += int(np.isneginf(feat).sum())
        feat = np.nan_to_num(feat, nan=NAN_FILL,
                             posinf=CLIP_ABS, neginf=-CLIP_ABS)
    NONFINITE_COUNTS["clipped"] += int((np.abs(feat) > CLIP_ABS).sum())
    return np.clip(feat, -CLIP_ABS, CLIP_ABS).astype(np.float32)


def _warn_if_sanitised(name: str) -> None:
    imputed = (NONFINITE_COUNTS["nan"] + NONFINITE_COUNTS["posinf"]
               + NONFINITE_COUNTS["neginf"])
    if imputed:
        warnings.warn(
            f"{name}: {imputed} non-finite feature values were imputed "
            f"({NONFINITE_COUNTS['nan']} NaN -> {NAN_FILL}, "
            f"{NONFINITE_COUNTS['posinf']} +inf -> {CLIP_ABS}, "
            f"{NONFINITE_COUNTS['neginf']} -inf -> {-CLIP_ABS}). "
            "Any downstream metric built on this read is imputed data.",
            RuntimeWarning, stacklevel=2)


def _read_dataset(name: str, parquet: str, max_rows: int | None,
                  per_class_cap: int | None = None, seed: int = 42):
    """Stream the big parquet, keep only rows whose Dataset == name.

    Memory-safe pattern ported from the sibling ids-compression-benchmark
    streaming loader (docs/Complete_Script_Inventory.md -> src/data/):
    - read in batches, clip extreme values, cast to float32 per batch (never hold
      float64 for a 37M-row dataset -> ~11.8 GB);
    - if `per_class_cap` is set, keep a per-attack-class reservoir (uniform random
      subsample via the priority/random-key method) so RAM is bounded at roughly
      per_class_cap * n_classes rows and RARE families are kept whole while common
      ones are subsampled (preserves the rare-family signal).

    `max_rows` and `per_class_cap` are mutually exclusive; passing both raises.
    The reservoir path samples uniformly over the WHOLE stream, so there is no
    honest way to also honour a first-N early stop (see the ValueError below).

    Returns (feature_names, X float32 [N,F], attack list, label int array).
    """
    if per_class_cap is not None and max_rows is not None:
        raise ValueError(
            f"max_rows={max_rows} cannot be combined with "
            f"per_class_cap={per_class_cap}. The bounded path draws a uniform "
            "random sample over the entire stream; stopping early would silently "
            "degrade it to a prefix sample (the parquet is grouped, so a prefix is "
            "not class-representative), and truncating afterwards would destroy the "
            "per-class balance the cap exists to preserve. Pass exactly one: "
            "max_rows for a cheap test read, per_class_cap for a bounded real read.")
    _reset_nonfinite_counts()
    f = pq.ParquetFile(parquet)
    all_cols = [f.schema_arrow.field(i).name for i in range(len(f.schema_arrow))]
    feat_cols = [c for c in all_cols
                 if c not in IDENTIFIER_COLS and c not in META_COLS]
    need = feat_cols + META_COLS
    rng = np.random.default_rng(seed)

    def _batch_features(batch, mask):
        feat = np.column_stack([
            np.asarray(batch.column(c).to_numpy(zero_copy_only=False),
                       dtype=np.float64)
            for c in feat_cols
        ])[mask]
        # sanitise + clamp + downcast immediately: bounds memory (never hold float64
        # for a 37M-row dataset) and keeps NaN/inf out of GCP. The clip alone cannot
        # do the NaN half of that job; see _sanitise_features.
        return _sanitise_features(feat)

    if per_class_cap is None:
        # --- unbounded path (small datasets / tests): stream, no float64 hoarding ---
        X_parts, atk_parts, lab_parts = [], [], []
        n = 0
        for batch in f.iter_batches(batch_size=1_000_000, columns=need):
            ds = np.asarray(batch.column("Dataset").to_pylist())
            mask = ds == name
            if not mask.any():
                continue
            X_parts.append(_batch_features(batch, mask))
            atk_parts.append(np.asarray(batch.column("Attack").to_pylist())[mask])
            lab_parts.append(
                np.asarray(batch.column("Label").to_numpy(zero_copy_only=False),
                           dtype=np.int64)[mask])
            n += int(mask.sum())
            if max_rows and n >= max_rows:
                break
        X = np.concatenate(X_parts, axis=0)
        attacks = np.concatenate(atk_parts, axis=0)
        labels = np.concatenate(lab_parts, axis=0)
        if max_rows:
            X, attacks, labels = X[:max_rows], attacks[:max_rows], labels[:max_rows]
        _warn_if_sanitised(name)
        return feat_cols, X, list(attacks), labels.astype(np.int64)

    # --- bounded path: per-class random-key reservoir (uniform subsample) ---
    # res[cls] = {"key": [m], "X": [m,F], "y": [m]}; keep the cap smallest keys.
    res: dict = {}
    for batch in f.iter_batches(batch_size=1_000_000, columns=need):
        ds = np.asarray(batch.column("Dataset").to_pylist())
        mask = ds == name
        if not mask.any():
            continue
        Xb = _batch_features(batch, mask)
        atk = np.asarray(batch.column("Attack").to_pylist())[mask]
        yb = np.asarray(batch.column("Label").to_numpy(zero_copy_only=False),
                        dtype=np.int64)[mask]
        keyb = rng.random(len(Xb))
        for cls in np.unique(atk):
            cm = atk == cls
            r = res.setdefault(cls, {"key": np.empty(0), "X": None, "y": np.empty(0, np.int64)})
            k = np.concatenate([r["key"], keyb[cm]])
            X = Xb[cm] if r["X"] is None else np.concatenate([r["X"], Xb[cm]])
            y = np.concatenate([r["y"], yb[cm]])
            if len(k) > per_class_cap:                 # keep cap smallest keys
                keep = np.argpartition(k, per_class_cap)[:per_class_cap]
                k, X, y = k[keep], X[keep], y[keep]
            r["key"], r["X"], r["y"] = k, X, y
    X = np.concatenate([r["X"] for r in res.values()], axis=0)
    attacks = np.concatenate([np.full(len(r["y"]), c) for c, r in res.items()])
    labels = np.concatenate([r["y"] for r in res.values()])
    _warn_if_sanitised(name)
    return feat_cols, X, list(attacks), labels.astype(np.int64)


def _row_hashes(X: np.ndarray) -> np.ndarray:
    return np.array([hashlib.sha1(r.tobytes()).hexdigest() for r in X])


def row_hashes_fast(X: np.ndarray, chunk_rows: int = 500_000) -> np.ndarray:
    """64-bit content key per row, vectorised over columns (FNV-1a).

    Same purpose as `_row_hashes`, byte-exact row identity for leak checks, but
    O(F) numpy passes instead of one Python sha1 call per row, and 8 bytes of
    output per row instead of a 40-character string. The cross-dataset loader
    keys up to 8 million rows per cell across 60 cells on a 14 GB box, where the
    sha1 path costs minutes of wall clock and hundreds of MB of Python strings.

    Collision bound: for n rows the birthday probability is about n**2 / 2**65.
    At n = 8e6 that is 1.7e-6. Report it alongside any duplicate count derived
    from this function.

    Note the float64 view means -0.0 and 0.0 key differently, and two NaNs with
    different payloads key differently. That is also true of `_row_hashes`,
    which hashes the same bytes, so the two agree.

    Keys are computed in row blocks of `chunk_rows`. A float32 input has to be
    widened to float64 before the uint64 view, and doing that in one pass on the
    production array (5.1M rows by 39 columns) allocates a 1.6 GB temporary on a
    box with 14 GB total. Blocking caps that temporary at about 156 MB and
    leaves the result bit-identical, since every row is keyed independently.
    """
    A = np.ascontiguousarray(X)
    if A.ndim != 2:
        raise ValueError(f"row_hashes_fast expects a 2-D array, got shape {A.shape}")
    if chunk_rows < 1:
        raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
    out = np.empty(A.shape[0], dtype=np.uint64)
    prime = np.uint64(1099511628211)
    for start in range(0, A.shape[0], chunk_rows):
        blk = A[start:start + chunk_rows]
        U = np.ascontiguousarray(blk.astype(np.float64, copy=False)).view(np.uint64)
        h = np.full(len(blk), np.uint64(14695981039346656037), dtype=np.uint64)
        for j in range(U.shape[1]):
            h = (h ^ U[:, j]) * prime
        out[start:start + len(blk)] = h
    return out


def materialize_dataset(name: str, parquet: str = DEFAULT_PARQUET,
                        out_dir: str = DEFAULT_OUT, seed: int = 42,
                        ratios=(0.70, 0.15, 0.15), max_rows: int | None = None,
                        min_class: int = 2,
                        per_class_cap: int | None = None) -> dict:
    """Build and persist train/val/test for one dataset. Returns meta dict.

    For the large datasets (BoT-IoT 37.8M etc.) pass `per_class_cap` (e.g. 400_000)
    to bound memory via a stratified per-class reservoir while keeping rare
    families whole. `max_rows` (first-N early stop) is for tests only and cannot be
    combined with `per_class_cap`; `_read_dataset` rejects that pair.
    """
    assert name in DATASETS, f"unknown dataset {name}"
    feat_cols, X, attacks, y_bin = _read_dataset(name, parquet, max_rows,
                                                 per_class_cap=per_class_cap,
                                                 seed=seed)
    attacks = np.asarray(attacks)

    # --- drop classes too small to stratify across 3 splits ---
    counts = collections.Counter(attacks.tolist())
    keep_mask = np.array([counts[a] >= min_class for a in attacks])
    X, attacks, y_bin = X[keep_mask], attacks[keep_mask], y_bin[keep_mask]

    # multiclass int codes from native attack strings
    classes = sorted(set(attacks.tolist()))
    code = {c: i for i, c in enumerate(classes)}
    y_multi = np.array([code[a] for a in attacks], dtype=np.int64)

    # --- stratified 70/15/15 on native multiclass (BEFORE any dedup) ---
    test_frac = ratios[2]
    val_frac = ratios[1] / (ratios[0] + ratios[1])
    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=test_frac,
                                      stratify=y_multi, random_state=seed)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=val_frac,
                                      stratify=y_multi[tr_idx], random_state=seed)

    # --- LEAKAGE-ONLY dedup (decision 2026-06-27): keep within-train repeats as
    # real traffic-frequency signal; remove only val/test rows whose feature-hash
    # appears in train (and test rows duplicating val). Splits become hash-disjoint
    # across boundaries with no train data loss. ---
    h = _row_hashes(X)
    n_raw = len(X)
    train_hashes = set(h[tr_idx].tolist())
    va_idx = va_idx[~np.isin(h[va_idx], list(train_hashes))]
    val_hashes = set(h[va_idx].tolist())
    block = train_hashes | val_hashes
    te_idx = te_idx[~np.isin(h[te_idx], list(block))]
    n_leak_removed = n_raw - (len(tr_idx) + len(va_idx) + len(te_idx))

    # --- drop constant columns (fit on TRAIN only) ---
    train_std = X[tr_idx].std(axis=0)
    nonconst = train_std > 0
    feat_cols = [c for c, k in zip(feat_cols, nonconst) if k]
    X = X[:, nonconst]

    # --- z-score fit on TRAIN only ---
    mu = X[tr_idx].mean(axis=0)
    sd = X[tr_idx].std(axis=0)
    sd[sd == 0] = 1.0
    Xz = ((X - mu) / sd).astype(np.float32)

    out = os.path.join(out_dir, name)
    os.makedirs(out, exist_ok=True)
    checks = {}
    for split, sidx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        np.save(os.path.join(out, f"{split}_X.npy"), Xz[sidx])
        np.save(os.path.join(out, f"{split}_ybin.npy"), y_bin[sidx])
        np.save(os.path.join(out, f"{split}_ymulti.npy"), y_multi[sidx])
        checks[f"{split}_X"] = hashlib.sha256(Xz[sidx].tobytes()).hexdigest()[:16]

    # n_raw counts the rows READ (after the min-class filter) and is therefore the
    # PRE-dedup figure; n_total counts the rows actually PERSISTED. They differ by
    # n_leakage_removed, on BoT-IoT by 41 percent, so reporting n_raw as the dataset
    # size overstates every persisted split.
    n_persisted = int(len(tr_idx) + len(va_idx) + len(te_idx))
    meta = {
        "dataset": name, "seed": seed, "n_features": len(feat_cols),
        "features": feat_cols, "classes": classes,
        "dedup_policy": "leakage-only",
        "n_raw": int(n_raw),
        "n_total": n_persisted,
        "n_leakage_removed": int(n_leak_removed),
        "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "sanitisation": dict(NONFINITE_COUNTS),
        "standardiser": {"mu": mu.tolist(), "sd": sd.tolist()},
        "checksums": checks,
    }
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


_CACHE: dict = {}


def load_split(name: str, split: str, out_dir: str = DEFAULT_OUT,
               label_mode: str = "binary"):
    """Load a cached split. Cached in-process so repeated calls don't re-read."""
    key = (name, split, out_dir)
    if key not in _CACHE:
        base = os.path.join(out_dir, name)
        _CACHE[key] = {
            "X": np.load(os.path.join(base, f"{split}_X.npy")),
            "binary": np.load(os.path.join(base, f"{split}_ybin.npy")),
            "multiclass": np.load(os.path.join(base, f"{split}_ymulti.npy")),
        }
    d = _CACHE[key]
    return d["X"], d[label_mode]
