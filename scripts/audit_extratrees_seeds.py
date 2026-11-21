#!/usr/bin/env python3
"""Audit item B12: are the Extra Trees per-seed results really seed-dependent?

Reads results/extratrees/*.json and reports, per dataset, the per-seed test and
validation macro-F1, the number of distinct values, the stored n_estimators, and
whether the summary file agrees with the per-seed files. Read-only.
Run from the repository root.
"""
import glob
import json
import os

DATASETS = ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v2"]
SEEDS = [42, 123, 456, 789, 1337]
summ = json.load(open("results/extratrees/extratrees_summary.json"))
for ds in DATASETS:
    rows = []
    for s in SEEDS:
        p = f"results/extratrees/{ds}__extratrees__binary__seed{s}.json"
        d = json.load(open(p))
        rows.append((s, d["macro_f1"], d["val_macro_f1"], d["n_test"], d["n_estimators"],
                     d["fit_predict_seconds"], os.path.getmtime(p)))
    t = [r[1] for r in rows]
    v = [r[2] for r in rows]
    print(ds)
    for r in rows:
        print(f"   seed {r[0]:5d}  test={r[1]:.6f}  val={r[2]:.6f}  n_test={r[3]}  trees={r[4]}  fit_s={r[5]}")
    print(f"   distinct test values={len(set(t))}  distinct val values={len(set(v))}")
    sp = summ["datasets"].get(ds, {}).get("per_seed")
    print(f"   summary per_seed matches files: {sp == t}")
