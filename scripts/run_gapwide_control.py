#!/usr/bin/env python3
"""Phase 0.1 -- capacity control for the GAP-vs-GCP comparison.

Methods claims the classifier is held fixed across pooling heads so any
difference is attributable to pooling. It is not: at C=32 the classifier reads
32 inputs under GAP and 528 under GCP, which is 20.0 percent more total
parameters at 2 classes and 67.7 percent more at 7. This runs a third head,
gap_wide, whose Linear(32, 528) projection gives GAP the same classifier width
(and strictly more total parameters than GCP), on the 5 canonical seeds.

If GCP still wins against gap_wide, the capacity confound is answered in one
table row and nothing else needs retraining.

Resumable: skips any (head, seed) whose result JSON already exists.
Cost: about 1.0 GPU-hour on an RTX 3060 Ti.

The --heads and --seeds flags exist so two processes can share one card without
racing the same cell. Two concurrent copies of a driver with no partition flag
exhausted RAM and froze this machine on 2026-07-06. Splitting seeds across two
processes on the SAME card is sound (identical hardware, identical kernels);
splitting them across two different GPUs is not, and is why the gap1 sweep
partitions by held-out dataset instead.
"""
from __future__ import annotations

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine import train_model, evaluate_model   # noqa: E402

DATASET = "NF-UNSW-NB15-v2"
HEADS = ["gap", "gap_wide", "gcp"]
SEEDS = [42, 123, 456, 789, 1337]
CKPT_DIR = "checkpoints/gapwide"
RESULTS_DIR = "results/gapwide"


def run(heads=None, seeds=None, dataset=DATASET, epochs=20,
        ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR):
    heads = list(heads) if heads else HEADS
    seeds = list(seeds) if seeds else SEEDS
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    total = len(heads) * len(seeds)
    done = 0
    for head in heads:
        for seed in seeds:
            tag = f"{dataset}__{head}__binary__seed{seed}"
            res_path = os.path.join(results_dir, f"{tag}.json")
            done += 1
            if os.path.exists(res_path):
                print(f"[{done}/{total}] {tag} SKIP (exists)", flush=True)
                continue
            _, ckpt_path = train_model(
                dataset, head, seed=seed, label_mode="binary",
                epochs=epochs, ckpt_dir=ckpt_dir,
            )
            res = evaluate_model(ckpt_path, dataset, label_mode="binary",
                                 results_dir=results_dir)
            print(f"[{done}/{total}] {tag} macro_f1={res['macro_f1']:.4f}",
                  flush=True)
    print("gapwide control done")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Phase 0.1 capacity control: GAP vs GAP-wide vs GCP")
    ap.add_argument("--heads", nargs="+", default=HEADS, choices=HEADS,
                    metavar="HEAD",
                    help="pooling heads this process owns")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                    metavar="SEED",
                    help="seeds this process owns; partition these across two "
                         "processes to share one card without racing a cell")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    args = ap.parse_args(argv)
    run(heads=args.heads, seeds=args.seeds, epochs=args.epochs,
        ckpt_dir=args.ckpt_dir, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
