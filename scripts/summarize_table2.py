#!/usr/bin/env python3
"""Recompute Table II from the per-seed result JSONs, with std and Holm.

Provenance
----------
Table II is assembled across four drivers. The map below was verified by
checking which directory reproduces the published means to four decimals, not by
trusting any summary file, because two summaries on disk are stale about it:

  results/maxpool/maxpool_summary.json declares NF-BoT-IoT-v2 provenance as
  results/gap4 and its per_head values match gap4, but the published table
  reproduces from results/botredo (gap4 gives 0.9846/0.9885/0.9832/0.9884
  against the table's 0.9835/0.9889/0.9835/0.9887). results/botredo is the
  deliberate same-driver rerun described in scripts/run_botiot_same_driver.py,
  which exists precisely to put NF-BoT-IoT-v2 on the same call path as the other
  corpora. It is the correct source. iccit/README.md line 24 is stale the same
  way.

Never glob results/*/ for these cells. Both results/gap4 and results/botredo hold
NF-BoT-IoT-v2 cells for all four heads and they disagree in every one.

Multiplicity
------------
Holm-Bonferroni is applied within each contrast's own four-corpus family, so four
tests per contrast rather than the roughly sixty verdicts the paper reports
overall. The p feeding it comes from the paired t = d_z * sqrt(n) on n - 1
degrees of freedom, which is an approximation at n = 5. Report it as a family-wise
check secondary to the bootstrap interval, which stays the primary evidence.
"""
from __future__ import annotations

import json
import math
import os
import statistics as st

from scipy.stats import t as tdist

DATASETS = ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2",
            "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v2"]
HEADS = ["gap", "gmp", "se", "gcp"]
SEEDS = [42, 123, 456, 789, 1337]
CONTRASTS = [("GCP-GMP", "gcp", "gmp"),
             ("GCP-GAP", "gcp", "gap"),
             ("GCP-SE", "gcp", "se")]

# dataset -> head -> results directory. Verified against the published table.
PROVENANCE = {
    "NF-UNSW-NB15-v2": {"gap": "repro", "se": "repro", "gcp": "repro",
                        "gmp": "maxpool", "hgb": "tree_baseline"},
    "NF-BoT-IoT-v2": {"gap": "botredo", "se": "botredo", "gcp": "botredo",
                      "gmp": "botredo", "hgb": "tree_baseline"},
    "NF-CSE-CIC-IDS2018-v2": {"gap": "gateC", "se": "gateC", "gcp": "gateC",
                              "gmp": "maxpool", "hgb": "tree_baseline"},
    "NF-ToN-IoT-v2": {"gap": "gateC", "se": "gateC", "gcp": "gateC",
                      "gmp": "maxpool", "hgb": "tree_baseline"},
}

# The means as printed in the current manuscript. The script refuses to emit a
# table body that does not reproduce these, so a silent source swap cannot pass.
PUBLISHED = {
    "NF-UNSW-NB15-v2": {"gap": 0.9390, "gmp": 0.9431, "se": 0.9404,
                        "gcp": 0.9436, "hgb": 0.9508},
    "NF-BoT-IoT-v2": {"gap": 0.9835, "gmp": 0.9889, "se": 0.9835,
                      "gcp": 0.9887, "hgb": 0.9937},
    "NF-CSE-CIC-IDS2018-v2": {"gap": 0.8779, "gmp": 0.8781, "se": 0.8784,
                              "gcp": 0.8783, "hgb": 0.8806},
    "NF-ToN-IoT-v2": {"gap": 0.9268, "gmp": 0.9424, "se": 0.9349,
                      "gcp": 0.9427, "hgb": 0.9652},
}


def load_per_seed(dataset, head):
    """Per-seed macro-F1 for one cell, in SEEDS order, from its own directory."""
    d = PROVENANCE[dataset][head]
    out = []
    for s in SEEDS:
        p = os.path.join("results", d,
                         f"{dataset}__{head}__binary__seed{s}.json")
        with open(p) as fh:
            out.append(json.load(fh)["macro_f1"])
    return out


def holm(pvals, alpha=0.05):
    """Holm-Bonferroni survival flags, returned in the input order."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, m = [False] * len(pvals), len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            out[i] = True
        else:
            break                      # Holm stops at the first failure
    return out


def main():
    stats = {ds: {h: load_per_seed(ds, h) for h in HEADS + ["hgb"]}
             for ds in DATASETS}

    print("== per-cell mean and standard deviation, n=5 ==")
    drift = []
    for ds in DATASETS:
        parts = []
        for h in HEADS + ["hgb"]:
            m, sd = st.mean(stats[ds][h]), st.stdev(stats[ds][h])
            parts.append(f"{h}={m:.4f}+-{sd:.4f}")
            if abs(round(m, 4) - PUBLISHED[ds][h]) > 1e-9:
                drift.append((ds, h, round(m, 4), PUBLISHED[ds][h],
                              PROVENANCE[ds][h]))
        print(f"{ds:22} {'  '.join(parts)}")
    if drift:
        print()
        print("MEANS DO NOT REPRODUCE THE PUBLISHED TABLE, refusing to emit a body:")
        for ds, h, got, want, d in drift:
            print(f"  {ds} {h}: got {got} from results/{d}, table says {want}")
        raise SystemExit(1)

    print()
    print("== paired contrasts, Holm within each contrast's four-corpus family ==")
    verdicts = {}
    for name, a, b in CONTRASTS:
        family = []
        for ds in DATASETS:
            per = [x - y for x, y in zip(stats[ds][a], stats[ds][b])]
            mu, sd = st.mean(per), st.stdev(per)
            dz = mu / sd if sd else float("nan")
            tstat = dz * math.sqrt(len(SEEDS))
            p = 2 * tdist.sf(abs(tstat), df=len(SEEDS) - 1)
            family.append((ds, mu, dz, tstat, p))
        for (ds, mu, dz, tstat, p), s in zip(family, holm([f[4] for f in family])):
            verdicts[(name, ds)] = (mu, dz, s)
            print(f"{name:8} {ds:22} mean={mu:+.4f} d_z={dz:+.3f} "
                  f"t={tstat:+.3f} p={p:.4f} holm={'survives' if s else '--'}")
        print()

    print("== Table II body, paste-ready ==")
    for ds in DATASETS:
        best = max(HEADS, key=lambda h: st.mean(stats[ds][h]))
        cells = []
        for h in HEADS + ["hgb"]:
            m, sd = st.mean(stats[ds][h]), st.stdev(stats[ds][h])
            cells.append(f"\\textbf{{{m:.4f}}} $\\pm$ {sd:.4f}" if h == best
                         else f"{m:.4f} $\\pm$ {sd:.4f}")
        mu, dz, s = verdicts[("GCP-GMP", ds)]
        cells += [f"{mu:+.4f}", f"{dz:+.2f}", "survives" if s else "--"]
        print(f"    {ds:22} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
