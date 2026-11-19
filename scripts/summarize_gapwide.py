#!/usr/bin/env python3
"""Summarise the Phase 0.1 capacity control on NF-UNSW-NB15-v2.

Methods claims the classifier is held fixed across pooling heads, so any
difference is attributable to pooling. It is not: at C=32 the classifier reads
32 inputs under GAP and C(C+1)/2 = 528 under GCP. `gap_wide` inserts a
Linear(32, 528) so GAP feeds the classifier the same width, which also gives it
strictly more total parameters than GCP.

Reports every pairwise contrast with a bootstrap CI, a t(df=4) CI and d_z, plus
a bit-identity check of the gap and gcp arms against the published run in
results/repro/t6b_summary.json.

Per src/metrics.py the t interval carries the "excludes zero" verdict at n=5;
the bootstrap is reported for comparability with the published numbers.

Writes results/gapwide/gapwide_summary.json and prints a table.
"""
from __future__ import annotations

import os
import sys
import json
import glob
import math

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import bootstrap_ci, t_ci, cohen_dz   # noqa: E402
from src.models import NIDSModel, count_params         # noqa: E402

RES_DIR = "results/gapwide"
PRIOR = "results/repro/t6b_summary.json"
OUT = os.path.join(RES_DIR, "gapwide_summary.json")
DATASET = "NF-UNSW-NB15-v2"
HEADS = ("gap", "gap_wide", "gcp")
DZ_GATE = 0.8


def _load():
    out = {h: {} for h in HEADS}
    for path in glob.glob(os.path.join(RES_DIR, f"{DATASET}__*__binary__seed*.json")):
        base = os.path.basename(path)[:-5]
        if "summary" in base:
            continue
        _, head, _, seed = base.split("__")
        out[head][int(seed[4:])] = float(json.load(open(path))["macro_f1"])
    return out


def _params():
    p = {}
    for h in HEADS:
        m = NIDSModel(h, 2, in_features=39)
        p[h] = {"head_out_dim": int(m.head.out_dim),
                "classifier_params": int(sum(x.numel()
                                             for x in m.classifier.parameters())),
                "total_params": int(count_params(m))}
    return p


def _contrast(a_name, a, b_name, b, seeds):
    av = np.array([a[s] for s in seeds])
    bv = np.array([b[s] for s in seeds])
    d = av - bv
    dmean, blo, bhi = bootstrap_ci(d)
    _, tlo, thi = t_ci(d)
    dz = cohen_dz(av, bv)
    return {"contrast": f"{a_name} - {b_name}",
            "per_seed_delta": {str(s): float(x) for s, x in zip(seeds, d)},
            "delta_mean": float(dmean),
            "delta_bootstrap_ci": [blo, bhi],
            "delta_t_ci": [tlo, thi],
            "d_z": dz if math.isfinite(dz) else None,
            "n_seeds_positive": int((d > 0).sum()),
            "t_ci_excludes_zero": bool(tlo > 0 or thi < 0),
            "bootstrap_ci_excludes_zero": bool(blo > 0 or bhi < 0),
            "passes_dz_gate": bool(abs(dz) >= DZ_GATE)}


def main():
    cells = _load()
    seeds = sorted(set.intersection(*(set(cells[h]) for h in HEADS)))
    if len(seeds) < 5:
        print(f"WARNING: only {len(seeds)} complete seeds: {seeds}")

    params = _params()
    summary = {"dataset": DATASET, "seeds": seeds, "label_mode": "binary",
               "params": params, "per_head": {}, "contrasts": [],
               "reproduction_check": {}}

    print(f"{'head':9} {'mean':>8} {'boot 95% CI':>20} {'clf par':>8} {'total par':>10}")
    print("-" * 60)
    for h in HEADS:
        v = [cells[h][s] for s in seeds]
        m, lo, hi = bootstrap_ci(v)
        _, tlo, thi = t_ci(v)
        summary["per_head"][h] = {
            "per_seed": {str(s): cells[h][s] for s in seeds},
            "mean": m, "bootstrap_ci": [lo, hi], "t_ci": [tlo, thi],
            "sd": float(np.std(v, ddof=1)), **params[h]}
        print(f"{h:9} {m:8.4f} [{lo:.4f},{hi:.4f}] {params[h]['classifier_params']:8d} "
              f"{params[h]['total_params']:10d}")

    print()
    hdr = (f"{'contrast':22} {'delta':>9} {'boot 95% CI':>20} "
           f"{'t(df=4) 95% CI':>20} {'d_z':>7} {'+/5':>4} {'t excl 0':>9}")
    print(hdr)
    print("-" * len(hdr))
    for a, b in (("gcp", "gap_wide"), ("gcp", "gap"), ("gap_wide", "gap")):
        c = _contrast(a, cells[a], b, cells[b], seeds)
        summary["contrasts"].append(c)
        dz = c["d_z"]
        print(f"{c['contrast']:22} {c['delta_mean']:+9.4f} "
              f"[{c['delta_bootstrap_ci'][0]:+.4f},{c['delta_bootstrap_ci'][1]:+.4f}] "
              f"[{c['delta_t_ci'][0]:+.4f},{c['delta_t_ci'][1]:+.4f}] "
              f"{dz:7.2f} {c['n_seeds_positive']:d}/5 "
              f"{str(c['t_ci_excludes_zero']):>9}")

    # ---- reproduction check against the published run ----
    if os.path.exists(PRIOR):
        prior = json.load(open(PRIOR))
        pseeds = prior["seeds"]
        for head, key in (("gap", "gap_macro_f1"), ("gcp", "gcp_macro_f1")):
            exact = all(cells[head][s] == prior[key][i]
                        for i, s in enumerate(pseeds) if s in cells[head])
            maxdiff = max(abs(cells[head][s] - prior[key][i])
                          for i, s in enumerate(pseeds) if s in cells[head])
            summary["reproduction_check"][head] = {
                "bit_identical": bool(exact), "max_abs_diff": float(maxdiff),
                "source": PRIOR}
            print(f"\nreproduction {head:4} vs {PRIOR}: "
                  f"bit-identical={exact}, max|diff|={maxdiff:.2e}")

    os.makedirs(RES_DIR, exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(summary, fh, indent=2)
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
