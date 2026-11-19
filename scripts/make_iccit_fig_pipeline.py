#!/usr/bin/env python3
"""ICCIT Fig. 1: the study pipeline as a full-width, stage-numbered diagram.

Rebuilt from the original single-column, generic-box version (small canvas,
4.8pt labels, boxes captioned only "One pooling head" / "Linear classifier")
into a two-column-spanning figure that names the real datasets, architecture,
and sweep sizes, matching the style used on the sibling NIDS/SAR papers' Fig. 1.

v2 (polish pass): swapped the cramped colour-coded column headers over the
chip grid for a proper swatch legend, gave the evaluation-grid container the
same rounded/filled style as every other box instead of a bare outline, eased
vertical spacing so nothing sits flush against a border, and unified arrow
routing so the top row (data) and bottom row (evaluation arms) each read as
one left-to-right flow with a single connector between them.

Every count printed on the figure is stated in iccit/sections/02_methods.tex
and must be re-checked there if the protocol changes.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "outputs/figures/fig_pipeline.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Paul Tol bright, colourblind-safe -- matches the sibling papers' figures.
HEAD_COLOR = {"gap": "#4477AA", "gmp": "#EE6677", "se": "#AA3377", "gcp": "#228833"}
NEUTRAL_FC, NEUTRAL_EC = "#F7F7F7", "#8a8a8a"
GRID_FC = "#FBFBFB"
ARROW_COLOR = "#6e6e6e"

RC = {
    "font.size": 7, "figure.dpi": 400, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "pdf.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
}
plt.rcParams.update(RC)

fig, ax = plt.subplots(figsize=(10.2, 3.7))
ax.set_xlim(0, 101.2)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, title, sub=None, fc=NEUTRAL_FC, ec=NEUTRAL_EC, tfs=8.9, sfs=7.4, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.4",
                                fc=fc, ec=ec, lw=lw))
    cx = x + w / 2
    if sub:
        two = "\n" in title
        ax.text(cx, y + h * (0.80 if two else 0.74), title, ha="center", va="center", fontsize=tfs,
                 fontweight="bold", color="#111111", linespacing=1.15)
        ax.text(cx, y + h * (0.28 if two else 0.32), sub, ha="center", va="center", fontsize=sfs,
                 color="#333333", linespacing=1.3)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center", fontsize=tfs,
                 fontweight="bold", color="#111111", linespacing=1.25)


def arrow(x1, y1, x2, y2, color=ARROW_COLOR, lw=1.15, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8.5,
                                 lw=lw, color=color, shrinkA=2.0, shrinkB=2.0,
                                 connectionstyle=f"arc3,rad={rad}"))


def stage(x, y, n):
    ax.text(x, y, n, ha="center", va="center", fontsize=7.0, color="#FFFFFF",
            fontweight="bold", bbox=dict(boxstyle="circle,pad=0.28", fc="#5a5a5a", ec="none"),
            zorder=5)


def chip(x, y, w, h, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.32",
                                fc=color, ec="none"))


def swatch_legend(x, y, entries, fs=7.0, gap=8.6):
    """One row: a small square swatch followed by its label, evenly spaced."""
    cx = x
    for label, color in entries:
        ax.add_patch(FancyBboxPatch((cx, y - 0.9), 1.6, 1.6,
                                    boxstyle="round,pad=0.02,rounding_size=0.2",
                                    fc=color, ec="none"))
        ax.text(cx + 2.3, y, label, ha="left", va="center", fontsize=fs,
                color="#222222", fontweight="bold")
        cx += gap


# ================= Top row: data pipeline (stages 1-3) =================
TOP_Y, TOP_H = 63, 30

stage(3.2, 96.5, "1")
box(2, TOP_Y, 23, TOP_H, "4 NetFlow-v2\ncorpora",
    "UNSW-NB15, CSE-CIC-\nIDS2018, BoT-IoT,\nToN-IoT, 38-39 features")
arrow(25, TOP_Y + TOP_H / 2, 28.4, TOP_Y + TOP_H / 2)

stage(30.6, 96.5, "2")
box(28.4, TOP_Y, 23, TOP_H, "Split $+$ scale",
    "Drop IP/port, 70/15/15\nhash-dedup, z-score\n(train split only)")
arrow(51.4, TOP_Y + TOP_H / 2, 54.8, TOP_Y + TOP_H / 2)

stage(57.8, 96.5, "3")
box(54.8, TOP_Y, 25, TOP_H, "CNN, TCN or MLP\nbackbone $+$ one head",
    "16-32-32 conv, kernel 3\nGAP / GMP / SE / GCP\nlinear head, Adam, 20 ep")

# The model stage feeds each evaluation arm (4, 5, 6).
arrow(56.5, TOP_Y, 44.0, 55.2)
arrow(63.2, TOP_Y, 63.2, 55.2)
arrow(77.5, TOP_Y, 86.0, 55.2)

# ================= Bottom row: evaluation arms (stages 4-7) =================
BOT_Y, BOT_H = 3, 51

stage(6, 60, "4")
box(2, BOT_Y, 47, BOT_H, "", None, fc=GRID_FC, lw=1.0)
ax.text(25.5, BOT_Y + BOT_H - 5.2, "Same-dataset $\\times$ LOO evaluation grid",
        ha="center", va="center", fontsize=8.2, fontweight="bold", color="#111111")
ax.text(25.5, BOT_Y + BOT_H - 10.0,
        "4 datasets $\\times$ 4 heads $\\times$ 5 seeds = 80 runs per arm",
        ha="center", va="center", fontsize=7.0, color="#333333")

heads = ["gap", "gmp", "se", "gcp"]
head_labels = {"gap": "GAP", "gmp": "GMP", "se": "SE", "gcp": "GCP"}
cw, ch, gx, gy = 2.1, 3.0, 1.6, 0.9
gx0, gy0 = 8, BOT_H - 17.5 + BOT_Y
for r in range(4):  # dataset row
    for c, h in enumerate(heads):  # head column
        chip(gx0 + c * (cw + gx), gy0 - r * (ch + gy), cw, ch, HEAD_COLOR[h])
ax.text(gx0 + 4 * (cw + gx) + 1.2, gy0 - 1.5 * (ch + gy), "4 held-out\ndatasets\n(rows)",
        ha="left", va="center", fontsize=6.7, color="#333333", linespacing=1.3)

legend_entries = [(head_labels[h], HEAD_COLOR[h]) for h in heads]
swatch_legend(6, BOT_Y + 8.6, legend_entries, gap=9.5)
ax.text(25.5, BOT_Y + 3.8, "$+$ TCN and MLP repeats, 40-epoch retrain (headline corpus)",
        ha="center", va="center", fontsize=6.8, color="#333333")

arrow(49.4, BOT_Y + 11, 52.0, BOT_Y + 11)

stage(72.2, 57.0, "5")
box(52.4, 30, 21.6, 24, "Channel-width sweep",
    "$C\\in\\{16,24,...,128\\}$\n$\\times$ n=5 seeds\npaired GCP$-$GAP $\\Delta$")

stage(98.2, 57.0, "6")
box(77, 30, 23, 24, "Mechanism probe", tfs=8.2,
    sub="grad$\\times$act top-5 vs.\nlabel-shuffle, $n{=}5$\nsynthetic generators")

arrow(63.2, 30, 63.2, 25)
arrow(88.5, 30, 88.5, 25)

stage(55.6, 21.0, "7")
box(52.4, BOT_Y, 47.6, 22, "Statistics",
    "paired Student-$t$ 95% CI, 4 df, Holm per contrast\n"
    "resolved when $|d_z|>1.242$ at $n{=}5$")

fig.savefig(OUT, bbox_inches="tight", pad_inches=0.04)
print("wrote", OUT)
