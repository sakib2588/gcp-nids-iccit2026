#!/usr/bin/env python3
"""Task 1 (effective-rank refinement) -- intermediate channel widths.

Fills the gaps between the Tier 1 sweep points {16,32,64,128} with widths
{24,48,96} so the delta-vs-width curve can be traced for an interior optimum.
Unlike Tier 1 these widths have NO prior results, so BOTH heads (gap, gcp) are
trained fresh; 3 widths x 5 seeds x 2 heads = 30 runs. Results are written into
results/tier1/ with the same naming so the downstream analysis reads one dir.

Robust: per-cell try/except, engine resume=True, single GPU, paired append
(gap AND gcp appended inside one try so a partial failure never misaligns the
per-seed pairs).
"""
import os
import sys
import json
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine import train_model, evaluate_model   # noqa: E402

DATASET = "NF-UNSW-NB15-v2"
C_VALUES = [24, 48, 96]
SEEDS = [42, 123, 456, 789, 1337]
CKPT_DIR = "checkpoints/task1"
RESULTS_DIR = "results/tier1"


def _macro_f1(head, c, seed):
    # engine tags as ...__seed{seed}__C{c}; result JSON name matches that.
    local = os.path.join(RESULTS_DIR,
                         f"{DATASET}__{head}__binary__seed{seed}__C{c}.json")
    if os.path.exists(local):
        return json.load(open(local))["macro_f1"]
    ckpt = train_model(DATASET, head, seed=seed, label_mode="binary", epochs=20,
                       ckpt_dir=CKPT_DIR, c_last=c, resume=True)[1]
    return evaluate_model(ckpt, DATASET, "binary", results_dir=RESULTS_DIR)["macro_f1"]


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for c in C_VALUES:
        for seed in SEEDS:
            try:
                g_gap = _macro_f1("gap", c, seed)
                g_gcp = _macro_f1("gcp", c, seed)
                print(f"[C{c} seed{seed}] gap={g_gap:.4f} gcp={g_gcp:.4f} "
                      f"delta={g_gcp - g_gap:+.4f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[C{c} seed{seed}] FAILED: {type(e).__name__}: {e}",
                      flush=True)
                traceback.print_exc()
    print("TASK1_WIDTHS_DONE", flush=True)


if __name__ == "__main__":
    main()
