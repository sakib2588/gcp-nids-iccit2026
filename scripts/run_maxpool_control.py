#!/usr/bin/env python3
"""Tier 1 -- global max pooling (GMP) as a first-order control.

Supervisor request: the paper races GCP against the MEAN (GAP), against channel
reweighting on top of the mean (SE), and against a width-matched mean
(gap_wide). It never races it against the other textbook first-order operator,
the max. This driver adds that arm on the three datasets whose published
baselines came from src.engine.train_model.

NF-BoT-IoT-v2 is deliberately NOT run here. Its published GAP/SE/GCP cells were
written by scripts/run_gap4.py's own training loop (condition "none"), not by
train_model. The two paths are equivalent in substance -- same seeding, epochs,
optimizer, _loader, grad clip and best-val selection -- but "equivalent in
substance" is not something a paired 5-seed delta should have to rely on, so the
BoT-IoT GMP cells go through run_gap4 instead:

    python -c "import sys; sys.path.insert(0,'.'); \
from scripts.run_gap4 import run_gap4; \
run_gap4(datasets=['NF-BoT-IoT-v2'], heads=['gmp'], conditions=['none'])"

DRIFT CHECK (--drift-check, on by default): also retrains GAP seed 42 on
NF-UNSW-NB15-v2 into this results directory and compares it against the
published value in results/repro/. The baselines this arm is paired against were
written weeks ago; if the toolchain has moved since, every GMP-minus-baseline
delta silently absorbs that drift. One extra cell buys the evidence, and
scripts/summarize_maxpool.py reports the verdict.

Heads available: gmp (width-matched to GAP, the weight-bearing contrast) and
gmp_wide (width-matched to GCP, the capacity control -- UNSW only is enough).

Resumable: any (dataset, head, seed) whose result JSON exists is skipped.

The --heads/--seeds/--datasets flags exist so two processes can share one card
without racing the same cell. Two concurrent copies of a driver with no
partition flag exhausted RAM and froze this machine on 2026-07-06. Splitting
seeds across two processes on the SAME card is sound; splitting them across two
different GPUs is not.

Pre-registration: notes/20260811-decision-maxpool-control.md (written before the
first cell ran; H-MAX-1/2/3 and the refutation condition are fixed there).
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine import train_model, evaluate_model   # noqa: E402

# Only the corpora whose published baselines came from train_model. BoT-IoT is
# reachable via --datasets but is not a default for the reason in the docstring.
DATASETS = ["NF-UNSW-NB15-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v2"]
HEADS = ["gmp"]
ALL_HEADS = ["gmp", "gmp_wide", "gap", "gcp"]   # gap for the drift check, gcp for
                                                # the epoch-budget sensitivity cell
SEEDS = [42, 123, 456, 789, 1337]
CKPT_DIR = "checkpoints/maxpool"
RESULTS_DIR = "results/maxpool"
DRIFT_DATASET = "NF-UNSW-NB15-v2"
DRIFT_SEED = 42
PUBLISHED_UNSW = "results/repro/NF-UNSW-NB15-v2__gap__binary__seed42.json"


def run_cell(dataset, head, seed, epochs, ckpt_dir, results_dir):
    """Train + evaluate one cell unless its result JSON already exists.

    Returns (result_dict, seconds) with seconds None when the cell was skipped,
    so the caller can report a MEASURED rate instead of extrapolating from the
    gapwide docstring's UNSW-only figure.
    """
    tag = f"{dataset}__{head}__binary__seed{seed}"
    res_path = os.path.join(results_dir, f"{tag}.json")
    if os.path.exists(res_path):
        with open(res_path) as fh:
            return json.load(fh), None
    t0 = time.time()
    # resume=True gives PER-EPOCH snapshots (<tag>__resume.pt, removed once the
    # cell finishes) carrying model, optimizer, epoch, best_val and every RNG
    # state, so a kill mid-cell continues the same random stream instead of
    # replaying epoch 0. Without it the resume granularity is a whole cell, and
    # an overnight run interrupted at epoch 19 of 20 throws that cell away.
    _, ckpt_path = train_model(dataset, head, seed=seed, label_mode="binary",
                               epochs=epochs, ckpt_dir=ckpt_dir, resume=True)
    res = evaluate_model(ckpt_path, dataset, label_mode="binary",
                         results_dir=results_dir)
    return res, time.time() - t0


def drift_check(results_dir, published=PUBLISHED_UNSW):
    """Compare the locally retrained GAP cell against the published one.

    Returns None when either side is missing. Reports max|diff| rather than a
    pass/fail: a non-identical result does not invalidate the run, it means
    every delta must be reported with the drift caveat attached.
    """
    local = os.path.join(results_dir,
                         f"{DRIFT_DATASET}__gap__binary__seed{DRIFT_SEED}.json")
    if not (os.path.exists(local) and os.path.exists(published)):
        return None
    with open(local) as fh:
        a = json.load(fh)["macro_f1"]
    with open(published) as fh:
        b = json.load(fh)["macro_f1"]
    return {"local": local, "published": published,
            "local_macro_f1": a, "published_macro_f1": b,
            "bit_identical": bool(a == b), "abs_diff": abs(a - b)}


def run(datasets=None, heads=None, seeds=None, epochs=20,
        ckpt_dir=CKPT_DIR, results_dir=RESULTS_DIR, do_drift_check=True,
        time_one_first=False):
    datasets = list(datasets) if datasets else DATASETS
    heads = list(heads) if heads else HEADS
    seeds = list(seeds) if seeds else SEEDS
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    cells = [(d, h, s) for d in datasets for h in heads for s in seeds]
    if do_drift_check and DRIFT_SEED in seeds:
        # Appended, not prepended: the drift check is evidence about the
        # baselines, not a gate on starting work.
        #
        # Gated on owning DRIFT_SEED so two seed-partitioned processes sharing
        # one card do not both claim this cell. The resume check only protects a
        # COMPLETED cell, so without this gate the two would train it
        # concurrently and race the same checkpoint path.
        cells.append((DRIFT_DATASET, "gap", DRIFT_SEED))
    elif do_drift_check:
        print(f"[drift] not owned by this process (seed {DRIFT_SEED} not in "
              f"{seeds}); the process owning it runs the check", flush=True)

    total, rates = len(cells), {}
    for i, (dataset, head, seed) in enumerate(cells, 1):
        res, secs = run_cell(dataset, head, seed, epochs, ckpt_dir, results_dir)
        if secs is None:
            print(f"[{i}/{total}] {dataset}__{head}__seed{seed} SKIP (exists)",
                  flush=True)
            continue
        rates.setdefault(dataset, []).append(secs)
        print(f"[{i}/{total}] {dataset}__{head}__seed{seed} "
              f"macro_f1={res['macro_f1']:.4f} ({secs/60:.1f} min)", flush=True)
        # Stop after the first cell this process actually TRAINED, not after
        # index 1: on a resumed run cell 1 is usually a skip, and stopping there
        # would either measure nothing or, worse, fall through and run the whole
        # sweep when the user asked for one cell.
        if time_one_first:
            print(f"[timing] first trained cell took {secs/60:.1f} min; "
                  f"{total} cells projects to ~{secs*total/3600:.1f} GPU-h "
                  f"(dataset sizes differ, so this is an anchor, not a total). "
                  f"Record it in notes/20260811-decision-maxpool-control.md.",
                  flush=True)
            return

    for dataset, secs in rates.items():
        print(f"[rate] {dataset}: {sum(secs)/len(secs)/60:.1f} min/cell "
              f"over {len(secs)} measured cells", flush=True)

    if do_drift_check:
        d = drift_check(results_dir)
        if d is None:
            print("[drift] skipped: local or published GAP cell missing")
        else:
            print(f"[drift] GAP {DRIFT_DATASET} seed{DRIFT_SEED}: "
                  f"bit-identical={d['bit_identical']} "
                  f"|diff|={d['abs_diff']:.2e} vs {d['published']}", flush=True)
    print("maxpool control done")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Tier 1 max-pooling control: GMP against the published "
                    "GAP/SE/GCP baselines")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, metavar="DATASET",
                    help="corpora this process owns (BoT-IoT belongs to "
                         "run_gap4, see module docstring)")
    ap.add_argument("--heads", nargs="+", default=HEADS, choices=ALL_HEADS,
                    metavar="HEAD",
                    help="gmp (GAP-width, the real contrast) and/or gmp_wide "
                         "(GCP-width capacity control)")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS, metavar="SEED",
                    help="seeds this process owns; partition these across two "
                         "processes to share one card without racing a cell")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--no-drift-check", action="store_true",
                    help="skip the retrained GAP seed-42 comparison against "
                         "results/repro (not recommended: it is the only "
                         "evidence that the published baselines are still "
                         "reproducible on this toolchain)")
    ap.add_argument("--time-one", action="store_true",
                    help="train exactly one not-yet-done cell, print its wall "
                         "time, and stop; skips do not count as the one cell")
    args = ap.parse_args(argv)
    run(datasets=args.datasets, heads=args.heads, seeds=args.seeds,
        epochs=args.epochs, ckpt_dir=args.ckpt_dir,
        results_dir=args.results_dir,
        do_drift_check=not args.no_drift_check,
        time_one_first=args.time_one)


if __name__ == "__main__":
    main()
