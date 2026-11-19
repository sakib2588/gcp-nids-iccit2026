#!/usr/bin/env python3
"""M4 (peer-review revision) -- second-dataset mechanism probe on NF-ToN-IoT-v2.

ToN-IoT is used as a POSITIVE CONTROL: it has the largest in-distribution GCP
advantage (+0.0159) of the four real datasets, so it is the dataset most likely
to carry the covariance signal the probe looks for. This is stated openly in the
paper so the choice is not read as deck-stacking; a null result here would still
be reported.

Real gcp/gap checkpoints already exist in checkpoints/gateC/ (from the gateC
headline run). Only the label-shuffled GCP control is missing, so this script:
  1. trains 5 label-shuffled GCP models for ToN into checkpoints/gap3_shuffled/
     (same dir + __shuffled tag convention as the UNSW controls; no collision
     because the filename carries the dataset name),
  2. runs the gap3 channel-attribution concentration probe for ToN, pointing the
     real-checkpoint dir at checkpoints/gateC and writing to results/gap3_ton/
     so the UNSW results/gap3/gap3_summary.json is never overwritten.
Resumable: shuffled checkpoints that already exist are skipped.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine import train_model              # noqa: E402
from run_gap3 import run_gap3                    # noqa: E402

DATASET = "NF-ToN-IoT-v2"
SEEDS = [42, 123, 456, 789, 1337]
SHUFFLED_CKPT_DIR = "checkpoints/gap3_shuffled"


def main():
    os.makedirs(SHUFFLED_CKPT_DIR, exist_ok=True)
    # 1) label-shuffled GCP controls for ToN
    for s in SEEDS:
        ckpt = os.path.join(
            SHUFFLED_CKPT_DIR, f"{DATASET}__gcp__binary__seed{s}__shuffled.pt")
        if os.path.exists(ckpt):
            print(f"[shuffled] skip seed {s} (exists)", flush=True)
            continue
        val, path = train_model(DATASET, "gcp", seed=s,
                                ckpt_dir=SHUFFLED_CKPT_DIR, shuffle_labels=True)
        print(f"[shuffled] seed {s}: val_macro_f1={val:.4f} -> {path}", flush=True)

    # 2) probe, reusing the existing gateC real checkpoints
    out = run_gap3(dataset=DATASET, seeds=tuple(SEEDS),
                   real_ckpt_dir="checkpoints/gateC",
                   shuffled_ckpt_dir=SHUFFLED_CKPT_DIR,
                   results_dir="results/gap3_ton")
    print("\n=== M4 ToN probe H5 ===", flush=True)
    print(out["h5"], flush=True)


if __name__ == "__main__":
    main()
