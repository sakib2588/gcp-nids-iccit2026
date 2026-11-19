#!/usr/bin/env python3
"""Gap1 cross-dataset LOO summary (G1b + G1c).

Prior-insensitive OOD metrics (AUC, balanced-accuracy) are primary per Review
R1-7. The comparison of interest is the GCP-minus-GAP delta under identical
shift (cancels the shared prior-shift component). Descriptive across-dataset
ranking only, no Friedman/Wilcoxon at n=4 datasets (project protocol SS7).

Paired contrasts are built from a single seed list (seeds where BOTH heads
carry a non-null value), never from two independently filtered lists zipped
positionally. The reported n is the length of that list, not the file count.
"""
import os
import sys
import json
import glob
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import bootstrap_ci, cohen_dz  # noqa: E402

RESULTS_DIR = "results/gap1"
SUMMARY_NAME = "gap1_summary.json"
# Per-cell results only. run_gap1.py writes "LOO__test-{ds}__{head}__seed{n}.json".
# A bare "*.json" would also match this script's own output, gap1_summary.json,
# which has no "test" key and raises KeyError on the second run.
CELL_GLOB = "LOO__*.json"

METRICS = ("auc", "balanced_accuracy", "macro_f1", "f1_calibrated")
DELTA_METRICS = ("auc", "balanced_accuracy", "macro_f1")
# Paired contrasts, written as (a, b) and reported as a minus b.
CONTRASTS = (("gcp", "gap"), ("gcp", "gmp"), ("gmp", "gap"))


def load_cells(results_dir=RESULTS_DIR):
    """Load every per-cell LOO result JSON under results_dir."""
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, CELL_GLOB))):
        with open(path) as fh:
            rows.append(json.load(fh))
    return rows


def index_cells(rows):
    """held-out test dataset -> head -> {seed: row}."""
    by = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by[r["test"]][r["head"]][r["seed"]] = r
    return by


def single_series(cells, key):
    """Unpaired series for one head: (seeds, values) over non-null seeds."""
    seeds = [s for s in sorted(cells) if cells[s].get(key) is not None]
    return seeds, [cells[s][key] for s in seeds]


def paired_series(cells_a, cells_b, key):
    """Paired series over ONE seed list. Returns (seeds, vals_a, vals_b).

    A seed enters only if both heads hold a non-null value for `key`, so
    vals_a[i] and vals_b[i] are always the same seed. Filtering each head
    independently and zipping positionally would silently pair head A's
    seed 42 against head B's seed 123 whenever the two heads are missing
    different seeds, and an equal-length guard does not catch that.
    """
    seeds = [s for s in sorted(set(cells_a) & set(cells_b))
             if cells_a[s].get(key) is not None
             and cells_b[s].get(key) is not None]
    return (seeds,
            [cells_a[s][key] for s in seeds],
            [cells_b[s][key] for s in seeds])


def summarize(by):
    summary = {}
    for test_ds, heads in by.items():
        # File count per head, kept for provenance. It is NOT the n behind any
        # interval below; each CI carries its own "_n" and "_seeds".
        block = {"n_cells": {h: len(cells) for h, cells in heads.items()}}
        for metric in METRICS:
            for h in heads:
                seeds, vals = single_series(heads[h], metric)
                if not vals:
                    continue
                block[f"{h}_{metric}_ci"] = bootstrap_ci(vals)
                block[f"{h}_{metric}_n"] = len(vals)
                block[f"{h}_{metric}_seeds"] = seeds
        # Paired head contrasts under identical shift. GCP-minus-GAP is the
        # primary one. GCP-minus-GMP mirrors the in-distribution max-pooling
        # arm out of distribution, which is the contrast that decides whether
        # the in-distribution null also holds under shift.
        for a, b in CONTRASTS:
            if a not in heads or b not in heads:
                continue
            for metric in DELTA_METRICS:
                seeds, vals_b, vals_a = paired_series(heads[b], heads[a],
                                                      metric)
                if not seeds:
                    continue
                delta = [x - y for x, y in zip(vals_a, vals_b)]
                key = f"delta_{a}_minus_{b}_{metric}"
                block[f"{key}_ci"] = bootstrap_ci(delta)
                block[f"{key}_n"] = len(seeds)
                block[f"{key}_seeds"] = seeds
                block[f"{key}_dz"] = (
                    cohen_dz(vals_a, vals_b) if len(delta) >= 2 else None)
        summary[test_ds] = block
    return summary


def main(results_dir=RESULTS_DIR, out_path=None):
    rows = load_cells(results_dir)
    by = index_cells(rows)
    summary = summarize(by)
    os.makedirs(results_dir, exist_ok=True)
    out_path = out_path or os.path.join(results_dir, SUMMARY_NAME)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary, rows, by


if __name__ == "__main__":
    _summary, _rows, _by = main()
    print(json.dumps(_summary, indent=2))
    print(f"\n[gap1] {len(_rows)} cells across {len(_by)} held-out datasets.")
