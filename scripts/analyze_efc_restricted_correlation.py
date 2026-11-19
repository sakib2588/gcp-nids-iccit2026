#!/usr/bin/env python3
"""Trace the restricted-family Spearman quoted in the manuscript.

The text reports a Spearman rho of about 0.05 between the per-family coupling
contrast index and the per-family GCP-minus-GAP gain, restricted to families
with at least 80,000 flows. That number was reproducible from disk but was
stored in no JSON, so it could not be traced (audit item 3.10). This script
recomputes it and writes it out.

Inputs, all read from disk and none modified:
  results/efc/efc_gcp_correlation.json  (the 15 pooled family rows as published)
  results/data_inventory.json           (per-dataset per-family corpus counts)

Output:
  results/efc/efc_gcp_correlation_restricted.json

The size filter uses the corpus-level family counts in data_inventory.json, i.e.
the total number of flows of that family in the source parquet, not the number
of rows that survived the materialised split.
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CORR = os.path.join(ROOT, "results", "efc", "efc_gcp_correlation.json")
INVENTORY = os.path.join(ROOT, "results", "data_inventory.json")
OUT = os.path.join(ROOT, "results", "efc", "efc_gcp_correlation_restricted.json")
MIN_FLOWS = 80_000


def spearman(x, y):
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def main():
    corr = json.load(open(CORR))
    inv = json.load(open(INVENTORY))["datasets"]

    kept, dropped = [], []
    for r in corr["rows"]:
        n = inv[r["dataset"]]["attacks"][r["family"]]
        row = dict(r)
        row["n_flows_corpus"] = int(n)
        (kept if n >= MIN_FLOWS else dropped).append(row)

    x = [r["efc_index"] for r in kept]
    y = [r["gcp_gain"] for r in kept]
    rho = spearman(x, y)
    pearson = float(np.corrcoef(x, y)[0, 1])

    out = {
        "description": ("Spearman and Pearson between the per-family coupling "
                        "contrast index and the per-family GCP minus GAP "
                        "macro-F1 gain, restricted to families with at least "
                        f"{MIN_FLOWS} flows in the source corpus."),
        "source_correlation_file": "results/efc/efc_gcp_correlation.json",
        "source_counts_file": "results/data_inventory.json",
        "index_definition": ("as published: off/(off+diag) on block SUMS, i.e. "
                             "not normalised by block count. Rank based "
                             "statistics are invariant to that defect, so this "
                             "restricted rho is unchanged by the block-count "
                             "correction; see "
                             "results/efc/efc_gcp_correlation_corrected.json"),
        "min_flows": MIN_FLOWS,
        "n_families_all": len(corr["rows"]),
        "n_families_kept": len(kept),
        "n_families_dropped": len(dropped),
        "spearman_rho": rho,
        "pearson_r": pearson,
        "dropped_families": [
            {"dataset": r["dataset"], "family": r["family"],
             "n_flows_corpus": r["n_flows_corpus"]} for r in dropped
        ],
        "rows": kept,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"kept {len(kept)} / {len(corr['rows'])} families at >= {MIN_FLOWS} flows")
    print(f"  Spearman rho = {rho:.4f}   Pearson r = {pearson:.4f}")
    print("  dropped: " + ", ".join(
        f"{r['family']}({r['n_flows_corpus']})" for r in dropped))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
