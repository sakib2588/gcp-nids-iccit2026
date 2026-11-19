#!/usr/bin/env python3
"""Fig. 3 -- channel-width sweep, paired GCP-GAP delta, NF-UNSW-NB15-v2, n=5.

The HDSS account predicts the advantage GROWS as C passes L'=33, where the
covariance becomes ill-conditioned. Prose cannot show the absence of a trend;
this can. Bars are Student-t 95% intervals on the paired per-seed differences.
"""
import json, math, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as tdist

rows = json.load(open("results/tier1/width_sweep_for_figure.json"))
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 6.4, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.5, 1.2))
C, mu, err = [], [], []
for r in rows:
    per = [c - g for c, g in zip(r["gcp"], r["gap"])]
    m, sd = st.mean(per), st.stdev(per)
    C.append(r["C"]); mu.append(m)
    err.append(tdist.ppf(.975, 4) * sd / math.sqrt(5))
ax.axhline(0, color="0.55", lw=0.7, zorder=1)
ax.axvline(33, color="0.55", lw=0.7, ls="--", zorder=1)
ax.text(34, 0.0092, r"$L'=33$", fontsize=6.2, color="0.35", va="top")
ax.errorbar(C, mu, yerr=err, color="#0072B2", marker="o", ms=3.2, lw=1.0,
            capsize=1.8, elinewidth=0.7, zorder=3)
ax.set_xscale("log")
ax.minorticks_off()
ax.set_xticks(C)
ax.set_xticklabels([str(c) for c in C])
ax.set_xlabel(r"Final-layer channels $C$ (ill-conditioned past $L'$)")
ax.set_ylabel("GCP $-$ GAP\nmacro-F1", labelpad=2)
# Lower limit from the data, so no interval cap is clipped (C=24 reaches below -0.001).
ax.set_ylim(min(-0.001, min(m - e for m, e in zip(mu, err)) - 0.0005), 0.0105)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(width=0.6, length=2.5)
fig.tight_layout(pad=0.4)
import os
os.makedirs("outputs/figures", exist_ok=True)
fig.savefig("outputs/figures/fig_width_sweep.pdf", bbox_inches="tight", pad_inches=0.08)
print("wrote outputs/figures/fig_width_sweep.pdf")
