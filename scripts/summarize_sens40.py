#!/usr/bin/env python3
"""Epoch-budget sensitivity on NF-UNSW-NB15-v2, 20 epochs against 40.

The published cells are 20 epochs. results/sens40/ holds the same three heads
retrained at 40 epochs, five seeds, everything else fixed, written to a separate
directory because src/engine.py:231-244 puts neither epochs nor lr in the result
filename and would otherwise have skipped every cell as already done.

Answers the reviewer's question 2, whether the GCP-versus-GMP null survives a
change of training budget.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st

from scipy.stats import t as tdist

SEEDS = [42, 123, 456, 789, 1337]
DS = "NF-UNSW-NB15-v2"
HEADS = ["gap", "gmp", "gcp"]

# 20-epoch provenance, verified against the published Table II.
PUB_DIR = {"gap": "repro", "gcp": "repro", "gmp": "maxpool"}


def load20(head):
    return [json.load(open(
        f"results/{PUB_DIR[head]}/{DS}__{head}__binary__seed{s}.json"))["macro_f1"]
        for s in SEEDS]


def load40(head):
    return [json.load(open(
        f"results/sens40/{DS}__{head}__binary__seed{s}.json"))["macro_f1"]
        for s in SEEDS]


def paired(a, b):
    per = [x - y for x, y in zip(a, b)]
    mu, sd = st.mean(per), st.stdev(per)
    dz = mu / sd if sd else float("nan")
    tstat = dz * math.sqrt(len(SEEDS))
    p = 2 * tdist.sf(abs(tstat), df=len(SEEDS) - 1)
    return mu, sd, dz, p, per


def main():
    missing = [h for h in HEADS
               for s in SEEDS
               if not os.path.exists(f"results/sens40/{DS}__{h}__binary__seed{s}.json")]
    if missing:
        raise SystemExit(f"sens40 incomplete, missing cells for heads {sorted(set(missing))}")

    for label, loader in (("epochs 20", load20), ("epochs 40", load40)):
        cells = {h: loader(h) for h in HEADS}
        print(f"== {label} ==")
        for h in HEADS:
            print(f"  {h}: mean={st.mean(cells[h]):.4f} sd={st.stdev(cells[h]):.4f}")
        for name, a, b in (("GCP-GMP", "gcp", "gmp"),
                           ("GCP-GAP", "gcp", "gap"),
                           ("GMP-GAP", "gmp", "gap")):
            mu, sd, dz, p, per = paired(cells[a], cells[b])
            verdict = "resolved" if abs(dz) > 1.24 else "not resolved"
            print(f"  {name}: mean={mu:+.4f} sd={sd:.4f} d_z={dz:+.3f} p={p:.4f} "
                  f"({verdict} at the |d_z|>1.24 rule)")
            print(f"    per_seed={[round(v, 4) for v in per]}")
        print()


if __name__ == "__main__":
    main()
