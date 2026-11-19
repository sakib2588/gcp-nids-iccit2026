#!/usr/bin/env python3
"""Tier 2b Task 5: correlate the per-family coupling contrast index (an
independent inverse-Potts oracle; it is NOT an Energy-based Flow Classifier, see
src/coupling_contrast.py) against the per-attack-family GCP-GAP macro-F1 gain
from the multiclass runs. Falsifiable prediction: GCP should help more on
families whose attack signal lives in feature covariance (high index).

NOTE (audit 2.3 / 2.4, 2026-07-20): this script reads the PRE-FIX
*_family_coupling.json files, whose cov_signal_index is not normalised by block
count and is computed on a degenerate one-hot encoding. It is kept as-is so the
published results/efc/efc_gcp_correlation.json stays reproducible. For the
corrected readings see scripts/analyze_efc_corrected_correlation.py.

All numbers read from disk:
  results/efc/{ds}_family_coupling.json   (cov_signal_index, by family NAME)
  results/multiclass/{ds}__{head}__multiclass__seed*.json (per_family_f1, by id)
  data/processed/{ds}/meta.json  (classes[] -> id<->name)
Reported descriptively (small n); pooled across the two datasets.
"""
import json
import glob
import numpy as np

SEEDS = [42, 123, 456, 789, 1337]
DATASETS = ["NF-ToN-IoT-v2", "NF-CSE-CIC-IDS2018-v2"]


def classes(ds):
    return json.load(open(f"data/processed/{ds}/meta.json"))["classes"]


def per_family_gain(ds):
    """mean-over-seeds GCP - GAP per-family F1, keyed by class id (str)."""
    def stack(head):
        rows = []
        for s in SEEDS:
            d = json.load(open(f"results/multiclass/{ds}__{head}__multiclass__seed{s}.json"))
            rows.append(d["per_family_f1"])
        return rows
    gap, gcp = stack("gap"), stack("gcp")
    ids = gap[0].keys()
    gain = {}
    for cid in ids:
        g = np.mean([r[cid] for r in gap])
        c = np.mean([r[cid] for r in gcp])
        gain[cid] = c - g
    return gain


def bootstrap_pearson(x, y, n_boot=10000):
    x, y = np.asarray(x), np.asarray(y)
    r = np.corrcoef(x, y)[0, 1]
    n = len(x)
    idx = np.arange(n)
    boots = []
    rng = np.random.default_rng(20260711)
    for _ in range(n_boot):
        b = rng.choice(idx, n, replace=True)
        if np.std(x[b]) > 0 and np.std(y[b]) > 0:
            boots.append(np.corrcoef(x[b], y[b])[0, 1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(r), float(lo), float(hi)


def spearman(x, y):
    xr = np.argsort(np.argsort(x)); yr = np.argsort(np.argsort(y))
    return float(np.corrcoef(xr, yr)[0, 1])


def main():
    rows = []
    for ds in DATASETS:
        cls = classes(ds)
        efc = json.load(open(f"results/efc/{ds}_family_coupling.json"))
        benign = efc["benign_class"]
        fams = efc["families"]
        gain = per_family_gain(ds)
        for name in cls:
            if name == benign or name not in fams:
                continue
            cid = str(cls.index(name))
            rows.append({"dataset": ds, "family": name,
                         "efc_index": fams[name]["cov_signal_index"],
                         "gcp_gain": gain[cid]})
    x = [r["efc_index"] for r in rows]
    y = [r["gcp_gain"] for r in rows]
    r, lo, hi = bootstrap_pearson(x, y)
    rho = spearman(x, y)
    out = {"n_families": len(rows), "datasets": DATASETS,
           "pearson_r": r, "pearson_ci95": [lo, hi], "spearman_rho": rho,
           "rows": rows}
    with open("results/efc/efc_gcp_correlation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"n={len(rows)} families  Pearson r={r:.3f} 95%CI[{lo:.3f},{hi:.3f}]  "
          f"Spearman rho={rho:.3f}")
    for r_ in sorted(rows, key=lambda z: -z["efc_index"]):
        print(f"  {r_['dataset'][:12]:12s} {r_['family']:14s} "
              f"efc={r_['efc_index']:.3f}  gcp_gain={r_['gcp_gain']:+.4f}")


if __name__ == "__main__":
    main()
