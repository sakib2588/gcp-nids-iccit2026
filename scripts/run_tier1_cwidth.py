#!/usr/bin/env python3
"""Tier 1 (novelty upgrade) -- channel-width HDSS sweep on NF-UNSW-NB15-v2.

The HDSS account (Discussion) predicts GCP helps only when the covariance is
ill-conditioned, i.e. when channels C exceed the number of pooled positions
L' (here L'=33). We sweep the final-conv channel count C over {16,32,64,128}
(L'/C = 2.06, 1.03, 0.52, 0.26) and measure the paired GCP-GAP macro-F1 delta.
Prediction: the delta grows as C crosses L' into the HDSS regime.

Reuse (no recompute): GCP at C in {16,32,64} from results/gap6/, GAP at C=32
from results/repro/. Only the missing cells are trained: GAP at {16,64,128} and
GCP at 128 (20 runs). Robust: per-cell try/except, resume=True, single GPU.
"""
import os
import sys
import json
import glob
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine import train_model, evaluate_model   # noqa: E402
from src.metrics import bootstrap_ci, cohen_dz        # noqa: E402

DATASET = "NF-UNSW-NB15-v2"
C_VALUES = [16, 32, 64, 128]
HEADS = ["gap", "gcp"]
SEEDS = [42, 123, 456, 789, 1337]
CKPT_DIR = "checkpoints/tier1"
RESULTS_DIR = "results/tier1"


def _reuse_path(head, c, seed):
    """Existing result file for a cell already computed elsewhere, or None."""
    if head == "gcp" and c in (16, 32, 64):
        return f"results/gap6/{DATASET}__gcp__binary__C{c}__seed{seed}.json"
    if head == "gap" and c == 32:
        return f"results/repro/{DATASET}__gap__binary__seed{seed}.json"
    if head == "gcp" and c == 32:
        return f"results/repro/{DATASET}__gcp__binary__seed{seed}.json"
    return None


def _macro_f1(head, c, seed):
    # 1) already trained into tier1 dir?
    # NOTE: engine tags as ...__seed{seed}__C{c} (seed BEFORE C, engine.py L134),
    # so the cache lookup must match that order or resume retrains every cell.
    suffix = "" if c == 32 else f"__C{c}"
    local = os.path.join(RESULTS_DIR, f"{DATASET}__{head}__binary__seed{seed}{suffix}.json")
    if os.path.exists(local):
        return json.load(open(local))["macro_f1"]
    # 2) reuse from gap6/repro?
    src = _reuse_path(head, c, seed)
    if src and os.path.exists(src):
        return json.load(open(src))["macro_f1"]
    # 3) train it
    ckpt = train_model(DATASET, head, seed=seed, label_mode="binary", epochs=20,
                       ckpt_dir=CKPT_DIR, c_last=c, resume=True)[1]
    return evaluate_model(ckpt, DATASET, "binary", results_dir=RESULTS_DIR)["macro_f1"]


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {"dataset": DATASET, "lprime": 33, "per_c": {}}
    for c in C_VALUES:
        gap, gcp, done_seeds, failed_seeds = [], [], [], []
        for seed in SEEDS:
            try:
                # Paired: compute BOTH first, append atomically. A partial
                # failure (e.g. GCP C128 training dies) must not leave gap with
                # an unmatched element -- that would misalign the per-seed pairs
                # and crash cohen_dz on unequal lengths.
                g_gap = _macro_f1("gap", c, seed)
                g_gcp = _macro_f1("gcp", c, seed)
                gap.append(g_gap)
                gcp.append(g_gcp)
                done_seeds.append(seed)
                print(f"[C{c} seed{seed}] gap={g_gap:.4f} gcp={g_gcp:.4f}", flush=True)
            except Exception as e:  # noqa: BLE001
                failed_seeds.append(seed)
                print(f"[C{c} seed{seed}] FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
        n = min(len(gap), len(gcp))  # always equal now; kept as a guard
        if n == 0:
            continue
        delta = [gcp[i] - gap[i] for i in range(n)]
        # `seeds` records WHICH seed each list position belongs to. Downstream
        # consumers must look up by seed value, never by SEEDS.index(seed):
        # a dropped seed shifts every later position (audit defect 1.13).
        summary["per_c"][str(c)] = {
            "lprime_over_c": 33 / c,
            "gap_mean_ci": bootstrap_ci(gap), "gcp_mean_ci": bootstrap_ci(gcp),
            "delta_mean_ci": bootstrap_ci(delta),
            "cohen_dz": cohen_dz(gcp, gap) if n >= 2 else None,
            "gap": gap, "gcp": gcp, "seeds": done_seeds,
            "failed_seeds": failed_seeds, "n": n,
        }
    with open(os.path.join(RESULTS_DIR, "tier1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Tier 1 channel-width sweep (GCP - GAP delta vs C) ===", flush=True)
    for c in C_VALUES:
        s = summary["per_c"].get(str(c))
        if s:
            d = s["delta_mean_ci"]
            print(f"  C={c:3d} (L'/C={33/c:.2f})  delta={d[0]:+.4f} "
                  f"CI[{d[1]:+.4f},{d[2]:+.4f}]  dz={s['cohen_dz']}", flush=True)
    print("TIER1_CWIDTH_DONE", flush=True)


if __name__ == "__main__":
    main()
