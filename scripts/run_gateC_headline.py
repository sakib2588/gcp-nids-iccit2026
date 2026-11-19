"""T8 -- GATE C: headline in-distribution GAP-vs-GCP for the 2 real datasets
missing a same-dataset train/eval run (CSE-CIC-IDS2018, ToN-IoT).

UNSW and BoT-IoT headline numbers already exist on disk
(results/repro/t6b_summary.json, results/gap4/*__none__*). This script fills
the other two so Task 8 (effective-rank vs GCP-win predictor) can run across
all 4 real datasets. Resumable: skips any (dataset, head, seed) whose result
JSON already exists.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine import train_model, evaluate_model   # noqa: E402

DATASETS = ["NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v2"]
HEADS = ["gap", "se", "gcp"]
SEEDS = [42, 123, 456, 789, 1337]
CKPT_DIR = "checkpoints/gateC"
RESULTS_DIR = "results/gateC"


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    total = len(DATASETS) * len(HEADS) * len(SEEDS)
    done = 0
    for name in DATASETS:
        for head in HEADS:
            for seed in SEEDS:
                tag = f"{name}__{head}__binary__seed{seed}"
                res_path = os.path.join(RESULTS_DIR, f"{tag}.json")
                done += 1
                if os.path.exists(res_path):
                    print(f"[{done}/{total}] {tag} SKIP (exists)", flush=True)
                    continue
                _, ckpt_path = train_model(
                    name, head, seed=seed, label_mode="binary",
                    epochs=20, ckpt_dir=CKPT_DIR,
                )
                res = evaluate_model(ckpt_path, name, label_mode="binary",
                                     results_dir=RESULTS_DIR)
                print(f"[{done}/{total}] {tag} macro_f1={res['macro_f1']:.4f}",
                     flush=True)
    print("gateC headline done")


if __name__ == "__main__":
    main()
