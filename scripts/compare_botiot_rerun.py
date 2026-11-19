"""Compare the NF-BoT-IoT-v2 same-driver rerun against the original gap4 cells.

The rerun exists to answer one question: does the BoT-IoT row of Table II depend
on having come from the class-imbalance experiment's untreated arm, or does it
reproduce when run through the headline driver?

This script answers it three ways and says plainly whether any printed claim moves:

  1. per-cell old-vs-new deltas, against the seed-to-seed spread for scale
  2. per-head means, against the values printed in Table II
  3. the paired contrasts the paper reports, with Cohen's d_z, under the paper's
     own resolution rule (|d_z| > 1.2417 at n=5)

Runs on however many cells exist, so it is useful mid-run. Exit code is 0 when
all 20 cells are present and no printed claim changes, 1 otherwise.
"""
from __future__ import annotations

import json
import math
import os
import sys

SEEDS = [42, 123, 456, 789, 1337]
HEADS = ["gap", "gmp", "se", "gcp"]
DATASET = "NF-BoT-IoT-v2"
OLD_DIR = "results/gap4"
NEW_DIR = "results/botredo"

# What the manuscript's Table II prints for this corpus.
PAPER_TABLE_II = {"gap": 0.9846, "gmp": 0.9885, "se": 0.9832, "gcp": 0.9884}
T975_DF4 = 2.7764451051977987
RESOLVE_AT = T975_DF4 / math.sqrt(5)          # 1.2417: |d_z| above this excludes zero


def macro_f1(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    for k in j:
        if "macro" in k.lower() and "f1" in k.lower() and "val" not in k.lower():
            return j[k]
    return None


def old(head: str, seed: int) -> float | None:
    return macro_f1(f"{OLD_DIR}/{DATASET}__{head}__none__seed{seed}.json")


def new(head: str, seed: int) -> float | None:
    return macro_f1(f"{NEW_DIR}/{DATASET}__{head}__binary__seed{seed}.json")


def paired(a: dict, b: dict) -> tuple[float, float, bool]:
    d = [a[s] - b[s] for s in SEEDS]
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    dz = m / sd if sd else math.inf
    return m, dz, abs(dz) > RESOLVE_AT


def main() -> int:
    have = {h: {s: v for s in SEEDS if (v := new(h, s)) is not None} for h in HEADS}
    n_cells = sum(len(v) for v in have.values())
    print(f"cells present: {n_cells}/20\n")

    print("1. PER-CELL, old (gap4) vs new (same driver)")
    deltas = []
    for h in HEADS:
        for s in SEEDS:
            o, nv = old(h, s), new(h, s)
            if o is None or nv is None:
                continue
            deltas.append(nv - o)
            print(f"   {h:4} seed{s:<5} old={o:.4f}  new={nv:.4f}  delta={nv - o:+.5f}")
    if deltas:
        worst = max(deltas, key=abs)
        print(f"\n   largest |delta| = {abs(worst):.5f}")

    print("\n2. PER-HEAD MEANS vs Table II as printed")
    moved = []
    for h in HEADS:
        if len(have[h]) != len(SEEDS):
            print(f"   {h:4} incomplete ({len(have[h])}/5)")
            continue
        mean = sum(have[h].values()) / len(SEEDS)
        printed = PAPER_TABLE_II[h]
        same = f"{mean:.4f}" == f"{printed:.4f}"
        if not same:
            moved.append((h, printed, mean))
        print(f"   {h:4} printed={printed:.4f}  rerun={mean:.4f}  "
              f"{'unchanged at 4dp' if same else 'CHANGED'}")

    print("\n3. PAIRED CONTRASTS under the paper's rule (|d_z| > 1.2417 resolves)")
    contrasts = [("gcp", "gmp"), ("gmp", "gap"), ("gcp", "gap"), ("gcp", "se")]
    for a, b in contrasts:
        if len(have[a]) != len(SEEDS) or len(have[b]) != len(SEEDS):
            print(f"   {a}-{b}: incomplete")
            continue
        m, dz, res = paired(have[a], have[b])
        print(f"   {a}-{b}: mean={m:+.5f}  d_z={dz:+.2f}  "
              f"{'RESOLVED' if res else 'not resolved'}")

    print()
    if n_cells < 20:
        print(f"VERDICT: incomplete, {n_cells}/20 cells. Re-run when finished.")
        return 1
    if moved:
        print("VERDICT: Table II MOVES at 4dp -- the manuscript needs updating:")
        for h, p, m in moved:
            print(f"  {h}: {p:.4f} -> {m:.4f}")
        return 1
    print("VERDICT: 20/20 cells, Table II unchanged at 4dp. The gap4 provenance "
          "did not distort the printed row.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
