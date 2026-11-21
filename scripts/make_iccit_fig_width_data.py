#!/usr/bin/env python3
"""Data for ICCIT Fig. 2 (channel-width sweep), consumed by the pgfplots source
iccit/figures/src/fig_width_sweep_cr.tex.

Same numbers and interval method as the earlier matplotlib figure: paired
per-seed GCP - GAP differences on NF-UNSW-NB15-v2, n = 5, Student-t 95% interval
(t = 2.7764451052 at 4 degrees of freedom). Nothing is re-estimated here.
Run from the repository root.
"""
import json
import math
import statistics as st

T975_DF4 = 2.7764451051977987
rows = json.load(open("results/tier1/width_sweep_for_figure.json"))
lines = ["C mean err lo hi"]
for r in rows:
    per = [c - g for c, g in zip(r["gcp"], r["gap"])]
    m, sd = st.mean(per), st.stdev(per)
    e = T975_DF4 * sd / math.sqrt(len(per))
    lines.append(f"{r['C']} {m:.6f} {e:.6f} {m - e:.6f} {m + e:.6f}")
open("iccit/figures/src/width_sweep.dat", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
