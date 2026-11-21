#!/usr/bin/env python3
"""Backing file for the ICCIT effective-rank correlation (ledger items F04 and B7).

The paper prints Pearson r = -0.15 (n = 4) between the Roy-Vetterli effective rank of
the standardized training features and the paired GCP - GAP macro-F1 gain. The older
results/gateC/gateC_summary.json and its _tci sibling store r = -0.236, because their
NF-BoT-IoT-v2 delta (0.0038) comes from results/gap4. The published Table II reproduces
from results/botredo (scripts/summarize_table2.py, PROVENANCE), where the delta is 0.0052.

This script recomputes the four deltas from the per-seed result files with exactly that
provenance map, recomputes r, and writes a NEW sibling file. It never overwrites an
existing summary (the journal draft's figure script still reads the old one).

Effective ranks are carried over unchanged from gateC_summary.json: they are computed from
the standardized training features, which do not depend on which run produced a delta, and
the split data is not needed here. The carried values are recorded in the output.

Usage (repository root; needs scipy because it imports summarize_table2):
  uv run --with scipy python scripts/recompute_effective_rank_correlation.py
"""
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(__file__))
from summarize_table2 import DATASETS, PROVENANCE, SEEDS, load_per_seed  # noqa: E402

T975_DF4 = 2.7764451051977987
OLD = "results/gateC/gateC_summary.json"
OUT = "results/gateC/gateC_summary_botredo.json"
PRINTED_DELTAS = {"NF-UNSW-NB15-v2": 0.0046, "NF-BoT-IoT-v2": 0.0052,
                  "NF-CSE-CIC-IDS2018-v2": 0.0004, "NF-ToN-IoT-v2": 0.0159}  # Table II, GCP-GAP
PRINTED_R = -0.15

old = json.load(open(OLD))
eranks = old["eranks"]
deltas, ci, sources = {}, {}, {}
for ds in DATASETS:
    gcp, gap = load_per_seed(ds, "gcp"), load_per_seed(ds, "gap")
    per = [c - g for c, g in zip(gcp, gap)]
    m, sd = st.mean(per), st.stdev(per)
    e = T975_DF4 * sd / len(per) ** 0.5
    deltas[ds] = m
    ci[ds] = [m, m - e, m + e]
    sources[ds] = {"gcp": f"results/{PROVENANCE[ds]['gcp']}", "gap": f"results/{PROVENANCE[ds]['gap']}"}

r = st.correlation([eranks[d] for d in DATASETS], [deltas[d] for d in DATASETS])

# the printed numbers must be reproduced, or the script refuses to write
for ds, p in PRINTED_DELTAS.items():
    assert round(deltas[ds], 4) == p, (ds, deltas[ds], p)
assert round(r, 2) == PRINTED_R, r

payload = {
    "eranks": eranks,
    "deltas": deltas,
    "pearson_r": r,
    "delta_ci": ci,
    "delta_sources": sources,
    "previous": {"deltas": old["deltas"], "pearson_r": old["pearson_r"],
                 "differs_only_in": [d for d in DATASETS
                                     if abs(old["deltas"][d] - deltas[d]) > 1e-12]},
    "_interval_method": "Student-t, df = n-1, two-sided 95 percent, sd with ddof=1",
    "_supersedes": [OLD, "results/gateC/gateC_summary_tci.json (its BoT delta)"],
    "_generated_by": "scripts/recompute_effective_rank_correlation.py",
    "_reason": ("ledger F04/B7: the stored BoT delta 0.0038 is from results/gap4; the published "
                "Table II reproduces from results/botredo (delta 0.0052), which gives r = -0.149, "
                "the value printed in the ICCIT paper"),
    "_note": "eranks are carried over from gateC_summary.json (training-feature statistic, unchanged)",
}
if os.path.exists(OUT):
    sys.exit(f"{OUT} exists; refusing to overwrite")
tmp = OUT + ".tmp"
with open(tmp, "w") as fh:
    json.dump(payload, fh, indent=2)
os.replace(tmp, OUT)
print(json.dumps({"deltas": {k: round(v, 6) for k, v in deltas.items()},
                  "pearson_r": round(r, 6), "old_r": round(old["pearson_r"], 6),
                  "differs_only_in": payload["previous"]["differs_only_in"]}, indent=1))
