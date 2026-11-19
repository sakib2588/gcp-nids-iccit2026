#!/usr/bin/env python3
"""Downstream effect of the two coupling-contrast corrections (audit 2.3, 2.4).

Recomputes the per-family index versus per-family GCP-minus-GAP correlation
under three readings of the same underlying quantity, so the effect of each
correction is separable:

  published    off/(off+diag) on block SUMS, degenerate one-hot columns left in
               and patched with nan_to_num   [results/efc/*_family_coupling.json]
  block_norm   the same run, with the block-count normalisation applied
               analytically: off/(off + (F-1)*diag). Exact, no recomputation
               needed, because the fix is a closed-form transform of the stored
               sums.
  corrected    full recomputation with degenerate columns dropped and the bin
               count taken from the distinct quantile edges, then block-count
               normalised  [results/efc/*_family_coupling_corrected.json]

Writes results/efc/efc_gcp_correlation_corrected.json. Never touches the
published results/efc/efc_gcp_correlation.json or the published
*_family_coupling.json files.
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EFC_DIR = os.path.join(ROOT, "results", "efc")
CORR = os.path.join(EFC_DIR, "efc_gcp_correlation.json")
OUT = os.path.join(EFC_DIR, "efc_gcp_correlation_corrected.json")


def spearman(x, y):
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def main():
    corr = json.load(open(CORR))
    published = {}
    corrected = {}
    for ds in corr["datasets"]:
        published[ds] = json.load(
            open(os.path.join(EFC_DIR, f"{ds}_family_coupling.json")))
        corrected[ds] = json.load(
            open(os.path.join(EFC_DIR, f"{ds}_family_coupling_corrected.json")))

    rows = []
    for r in corr["rows"]:
        ds, fam = r["dataset"], r["family"]
        pub = published[ds]
        F = pub["n_features"]
        off_sum, diag_sum = pub["families"][fam]["off"], pub["families"][fam]["diag"]
        # published sums divided by their own block counts, F(F-1) and F
        off_m, diag_m = off_sum / (F * (F - 1)), diag_sum / F
        cor = corrected[ds]["families"][fam]
        rows.append({
            "dataset": ds,
            "family": fam,
            "gcp_gain": r["gcp_gain"],
            "index_published": r["efc_index"],
            "index_block_normalised": float(off_m / (off_m + diag_m)),
            "index_corrected": cor["coupling_contrast_index"],
            "index_corrected_unnormalised": float(
                cor["off_sum"] / (cor["off_sum"] + cor["diag_sum"])),
            "marginal_over_covariance_ratio_published": float(diag_m / off_m),
            "n_active_features_corrected": cor["n_active_features"],
            "n_active_columns_corrected": cor["n_active_columns"],
        })

    gain = [r["gcp_gain"] for r in rows]
    variants = {}
    for key in ("index_published", "index_block_normalised", "index_corrected",
                "index_corrected_unnormalised"):
        v = [r[key] for r in rows]
        variants[key] = {
            "pearson_r": float(np.corrcoef(v, gain)[0, 1]),
            "spearman_rho": spearman(v, gain),
            "min": float(min(v)),
            "max": float(max(v)),
        }
    variants["index_published"]["spearman_vs_block_normalised"] = spearman(
        [r["index_published"] for r in rows],
        [r["index_block_normalised"] for r in rows])
    variants["index_published"]["spearman_vs_corrected"] = spearman(
        [r["index_published"] for r in rows],
        [r["index_corrected"] for r in rows])

    # same three readings on the >= 80k flow subset the manuscript restricts to
    inv = json.load(open(os.path.join(ROOT, "results", "data_inventory.json")))["datasets"]
    big = [r for r in rows
           if inv[r["dataset"]]["attacks"][r["family"]] >= 80_000]
    big_gain = [r["gcp_gain"] for r in big]
    restricted = {"min_flows": 80_000, "n_families": len(big)}
    for key in ("index_published", "index_block_normalised", "index_corrected"):
        v = [r[key] for r in big]
        restricted[key] = {
            "pearson_r": float(np.corrcoef(v, big_gain)[0, 1]),
            "spearman_rho": spearman(v, big_gain),
        }

    out = {
        "description": ("Per-family coupling contrast index versus per-family "
                        "GCP minus GAP macro-F1 gain, under the published "
                        "index and under the two audit corrections. The "
                        "published files are left untouched."),
        "n_families": len(rows),
        "datasets": corr["datasets"],
        "variants": variants,
        "restricted_to_large_families": restricted,
        "rows": rows,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)

    for k, v in variants.items():
        print(f"{k:34s} pearson={v['pearson_r']:+.4f} spearman={v['spearman_rho']:+.4f} "
              f"range=[{v['min']:.4f}, {v['max']:.4f}]")
    print(f"Spearman(published, block_normalised) = "
          f"{variants['index_published']['spearman_vs_block_normalised']:.4f}")
    print(f"Spearman(published, corrected)        = "
          f"{variants['index_published']['spearman_vs_corrected']:.4f}")
    print(f"restricted to >= 80k flows, n={restricted['n_families']}:")
    for key in ("index_published", "index_block_normalised", "index_corrected"):
        print(f"  {key:28s} pearson={restricted[key]['pearson_r']:+.4f} "
              f"spearman={restricted[key]['spearman_rho']:+.4f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
