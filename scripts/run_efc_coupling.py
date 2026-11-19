"""Tier 2a driver: per-attack-family inverse-Potts coupling contrast.

For each multi-class dataset, fit an inverse-Potts coupling model on benign
train flows, then for every attack family measure how much of the family's
deviation-from-benign structure lives in cross-feature coupling (off-diagonal
blocks) versus single-feature marginal shift (diagonal blocks). Emits, per
family:
    {off, diag, coupling_contrast_index = off / (off + diag), n_rows, ...}
to results/efc/{dataset}_family_coupling_corrected.json.

off and diag are means PER BLOCK. There are F(F-1) off-diagonal blocks against
F diagonal ones, so the pre-2026-07-20 version of this driver, which summed
without dividing, inflated the off-diagonal side by F-1 and reported every
family as covariance dominated. The pre-fix outputs are kept unmodified at
results/efc/{dataset}_family_coupling.json for comparison; this driver writes to
a separate CORRECTED_SUFFIX file and never overwrites them.

coupling_contrast_index in [0, 1]: higher => more of the family's departure from
benign lives in feature co-occurrence (the regime GCP could exploit); lower =>
more of it is a marginal shift. The index is NOT calibrated so that 0.5
separates the two regimes; only the ordering across families is interpretable.

CPU-only, OOM-safe: the model caps the benign fit at 80k rows and each scored
family at 40k rows via a fixed rng(42) subsample (see src/coupling_contrast.py).
Benign class is resolved by NAME from meta.json (it is NOT always class id 0,
e.g. ToN-IoT sorts 'Benign' to id 1, behind 'Backdoor').
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import data as D                              # noqa: E402
from src.coupling_contrast import CouplingContrast     # noqa: E402

DATASETS = ["NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]
N_BINS = 30
RIDGE = 1e-3
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "efc")
CORRECTED_SUFFIX = "_family_coupling_corrected.json"


def _meta(dataset):
    with open(os.path.join(PROC_DIR, dataset, "meta.json")) as fh:
        return json.load(fh)


def run_dataset(dataset: str) -> dict:
    meta = _meta(dataset)
    classes = meta["classes"]                 # class id -> name (sorted)
    assert "Benign" in classes, f"no Benign class in {dataset}: {classes}"
    benign_id = classes.index("Benign")

    X, y = D.load_split(dataset, "train", label_mode="multiclass")
    y = np.asarray(y)
    print(f"[{dataset}] X={X.shape} classes={classes} benign_id={benign_id} "
          f"benign_n={int((y == benign_id).sum())}")

    cc = CouplingContrast(n_bins=N_BINS, ridge=RIDGE).fit(X[y == benign_id])
    eff_bins = cc.effective_bins_
    print(f"  requested n_bins={N_BINS}; realised bins/feature "
          f"min={min(eff_bins)} median={int(np.median(eff_bins))} "
          f"max={max(eff_bins)}; gauge-fixed columns={cc.n_columns_} "
          f"(nominal {meta['n_features'] * (N_BINS - 1)})")

    families = {}
    for cid, name in enumerate(classes):
        if cid == benign_id:
            continue
        rows = X[y == cid]
        if rows.shape[0] < 2:                  # cannot form a correlation
            print(f"  - {name}: only {rows.shape[0]} rows, skipped")
            continue
        d = cc.block_coupling_detail(rows)
        families[name] = {
            "off": d["off"],
            "diag": d["diag"],
            "coupling_contrast_index": d["index"],
            # raw block sums retained only so the pre-fix un-normalised ratio
            # stays reconstructible for the audit trail; do not interpret them
            "off_sum": d["off_sum"],
            "diag_sum": d["diag_sum"],
            "n_off_blocks": d["n_off_blocks"],
            "n_diag_blocks": d["n_diag_blocks"],
            "n_active_features": d["n_active_features"],
            "n_active_columns": d["n_active_columns"],
            "n_rows": int(rows.shape[0]),
        }
        print(f"  - {name:12s} off={d['off']:.6f} diag={d['diag']:.6f} "
              f"index={d['index']:.4f} (n={rows.shape[0]})")

    out = {
        "dataset": dataset,
        "n_features": meta["n_features"],
        "n_bins_requested": N_BINS,
        "effective_bins_per_feature": eff_bins,
        "n_gauge_fixed_columns": cc.n_columns_,
        "ridge": RIDGE,
        "benign_class": "Benign",
        "benign_id": benign_id,
        "fit_row_cap": cc.max_fit_rows,
        "score_row_cap": cc.max_score_rows,
        "subsample_seed": 42,
        "block_count_normalised": True,
        "degenerate_columns_dropped": True,
        "families": families,
    }
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for dataset in DATASETS:
        out = run_dataset(dataset)
        path = os.path.join(OUT_DIR, f"{dataset}{CORRECTED_SUFFIX}")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"[{dataset}] wrote {path}\n")


if __name__ == "__main__":
    main()
