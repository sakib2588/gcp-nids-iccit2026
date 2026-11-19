"""R2+ -- feature-order sensitivity of the GCP-GAP advantage.

WHY THIS EXISTS
The backbone presents a NetFlow record as a 1-channel sequence, so a kernel-3
convolution slides over ADJACENT COLUMNS of the parquet schema (src/models.py
:7-9, :222-225). NetFlow columns have no meaningful order -- the adjacency is
an artefact of the CSV/parquet layout. Every learned channel, the covariance
the GCP head pools, and therefore the entire +0.0046 headline delta, inherit
that arbitrary choice. Nothing in the paper varies it.

A reviewer's one-line version: "your effect may be a property of one column
ordering." This script measures whether it is.

DESIGN
N FIXED permutations, each trained under the same five seeds and both heads:
N x 5 x 2 runs. --perms defaults to 3 orderings here; scripts/run_tonight.sh
overrides it to 5, which is what the overnight sweep actually runs. The permutation is keyed on its own seed, never on
the training seed, so ordering variance and seed variance stay separable --
that separation is the whole point, and drawing a fresh permutation per seed
would destroy it. The published identity-order result (results/repro/) is a
free fourth ordering.

WHAT WOULD FALSIFY THE PAPER
If the paired GCP-GAP delta swings across orderings by more than it swings
across seeds, the headline effect is substantially an artefact of column order
and the paper must say so. The summary reports both spreads side by side so
that comparison cannot be avoided.

PRE-SPECIFIED READING (written before the run, per the paper's own stance on
forking paths): the delta is expected to stay positive and interval-clear of
zero under all three orderings. Anything else is reported as found.

Resumable: skips any (head, seed, perm) whose result JSON already exists, so
an interrupted overnight run continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.engine import train_model, evaluate_model, _atomic_json_dump  # noqa: E402
from src.metrics import t_ci, cohen_dz  # noqa: E402

DATASET = "NF-UNSW-NB15-v2"
HEADS = ["gap", "gcp"]
SEEDS = [42, 123, 456, 789, 1337]
PERMS = [1, 2, 3]
CKPT_DIR = "checkpoints/feature_perm"
RESULTS_DIR = "results/feature_perm"
# The identity-ordering run already on disk; used as the fourth ordering.
BASELINE_DIR = "results/repro"


def cell_tag(dataset, head, seed, perm):
    return f"{dataset}__{head}__binary__seed{seed}__perm{perm}"


def load_f1(path):
    with open(path) as f:
        return json.load(f)["macro_f1"]


def usable_result(path):
    """True only if the cell result exists AND parses AND carries macro_f1.

    A bare os.path.exists is not enough. A dump interrupted by a power cut or
    an OOM kill leaves a truncated file that exists, so the cell gets skipped
    forever and the sweep silently loses it. Writes are atomic now
    (src.engine._atomic_json_dump), but files written before that fix, or by
    another tool, can still be short -- so verify rather than assume.
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            return "macro_f1" in json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, TypeError) as exc:
        print(f"[warn] unreadable result {path} ({exc}); recomputing this cell",
              flush=True)
        return False


def collect_deltas(results_dir, dataset, seeds, perms):
    """Paired GCP-GAP delta per (perm, seed). Returns {perm: [delta per seed]}."""
    out = {}
    for p in perms:
        row = []
        for s in seeds:
            gap = os.path.join(results_dir, f"{cell_tag(dataset,'gap',s,p)}.json")
            gcp = os.path.join(results_dir, f"{cell_tag(dataset,'gcp',s,p)}.json")
            if not (usable_result(gap) and usable_result(gcp)):
                row = None
                break
            row.append(load_f1(gcp) - load_f1(gap))
        if row is not None:
            out[p] = row
    return out


def baseline_deltas(baseline_dir, dataset, seeds):
    """Identity-ordering deltas from the published run, if present."""
    row = []
    for s in seeds:
        gap = os.path.join(baseline_dir, f"{dataset}__gap__binary__seed{s}.json")
        gcp = os.path.join(baseline_dir, f"{dataset}__gcp__binary__seed{s}.json")
        if not (usable_result(gap) and usable_result(gcp)):
            return None
        row.append(load_f1(gcp) - load_f1(gap))
    return row


def summarise(per_perm, baseline):
    """Ordering spread vs seed spread -- the comparison the result turns on.

    Refuses n<2: t_ci at n=1 returns [-inf, +inf] and numpy emits a
    "Degrees of freedom <= 0" warning, which renders as a real-looking row
    ("COVERS 0") that means nothing. Observed during a single-seed probe.

    Such an ordering is DROPPED with a warning rather than raising: an
    interrupted overnight run must still yield a partial summary.
    """
    dropped = [k for k, r in per_perm.items() if len(r) < 2]
    for k in dropped:
        print(f"[warn] ordering perm{k} has n={len(per_perm[k])} seed(s); "
              "dropped, an interval needs n>=2", flush=True)
    per_perm = {k: r for k, r in per_perm.items() if len(r) >= 2}
    if baseline is not None and len(baseline) < 2:
        print(f"[warn] identity baseline has n={len(baseline)} seed(s); dropped",
              flush=True)
        baseline = None
    if not per_perm and baseline is None:
        return {"per_ordering": {}, "n_orderings": 0,
                "_incomplete": ("no ordering had n>=2 seeds; nothing to "
                                "summarise. Re-run to completion."),
                "_dropped_orderings": dropped}
    summary = {"per_ordering": {}}
    means = []
    if baseline is not None:
        m, lo, hi = t_ci(baseline)
        summary["per_ordering"]["identity"] = {
            "delta_t_ci": [m, lo, hi], "excludes_zero": bool(lo > 0 or hi < 0),
            "per_seed": baseline,
        }
        means.append(m)
    for p, row in sorted(per_perm.items()):
        m, lo, hi = t_ci(row)
        summary["per_ordering"][f"perm{p}"] = {
            "delta_t_ci": [m, lo, hi], "excludes_zero": bool(lo > 0 or hi < 0),
            "per_seed": row,
        }
        means.append(m)

    # Spread ACROSS orderings (of the per-ordering means) against the typical
    # spread ACROSS seeds within an ordering. If the former dominates, column
    # order is doing work the paper attributes to the pooling head.
    within = [float(np.std(r, ddof=1))
              for r in ([baseline] if baseline is not None else []) + list(per_perm.values())]
    summary["ordering_sd"] = float(np.std(means, ddof=1)) if len(means) > 1 else None
    summary["mean_within_ordering_seed_sd"] = float(np.mean(within)) if within else None
    if summary["ordering_sd"] is not None and summary["mean_within_ordering_seed_sd"]:
        summary["ordering_to_seed_sd_ratio"] = (
            summary["ordering_sd"] / summary["mean_within_ordering_seed_sd"])
    summary["n_orderings"] = len(means)
    summary["all_orderings_exclude_zero"] = all(
        v["excludes_zero"] for v in summary["per_ordering"].values())
    summary["_method"] = {
        "interval": "Student-t, df = n-1, two-sided 95 percent, sd with ddof=1",
        "why_not_bootstrap": (
            "percentile bootstrap of a mean at n=5 has 0.839 coverage, see "
            "src/metrics.py and notes/20260720-interval-comparison.md"),
        "delta": "paired GCP minus GAP test macro-F1 at matched (ordering, seed)",
        "permutation": "src.engine._feature_permutation(n_features, perm_seed)",
        "identity_source": BASELINE_DIR,
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--heads", nargs="+", default=HEADS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--perms", nargs="+", type=int, default=PERMS)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--baseline-dir", default=BASELINE_DIR)
    ap.add_argument("--summarise-only", action="store_true",
                    help="rebuild the summary from existing cells, train nothing")
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.summarise_only:
        total = len(args.perms) * len(args.heads) * len(args.seeds)
        done = 0
        for perm in args.perms:
            for head in args.heads:
                for seed in args.seeds:
                    tag = cell_tag(args.dataset, head, seed, perm)
                    res_path = os.path.join(args.out_dir, f"{tag}.json")
                    done += 1
                    if usable_result(res_path):
                        print(f"[{done}/{total}] {tag} SKIP (exists)", flush=True)
                        continue
                    _, ckpt_path = train_model(
                        args.dataset, head, seed=seed, label_mode="binary",
                        epochs=args.epochs, ckpt_dir=args.ckpt_dir,
                        permute_features=perm,
                    )
                    res = evaluate_model(ckpt_path, args.dataset,
                                         label_mode="binary",
                                         results_dir=args.out_dir)
                    print(f"[{done}/{total}] {tag} macro_f1={res['macro_f1']:.4f}",
                          flush=True)

    per_perm = collect_deltas(args.out_dir, args.dataset, args.seeds, args.perms)
    base = baseline_deltas(args.baseline_dir, args.dataset, args.seeds)
    summary = summarise(per_perm, base)
    out = os.path.join(args.out_dir, "feature_perm_summary.json")
    _atomic_json_dump(summary, out)

    print("\nPaired GCP-GAP delta by feature ordering "
          f"(n={len(args.seeds)} seeds, Student-t df={len(args.seeds)-1}):")
    for k, v in summary["per_ordering"].items():
        m, lo, hi = v["delta_t_ci"]
        mark = "excl 0" if v["excludes_zero"] else "COVERS 0"
        print(f"  {k:<10} {m:+.5f} [{lo:+.5f}, {hi:+.5f}]  {mark}")
    if summary.get("ordering_to_seed_sd_ratio") is not None:
        print(f"\n  sd across orderings      : {summary['ordering_sd']:.5f}")
        print(f"  mean sd across seeds     : {summary['mean_within_ordering_seed_sd']:.5f}")
        print(f"  ordering/seed sd ratio   : {summary['ordering_to_seed_sd_ratio']:.2f}"
              "   (>1 means column order moves the delta more than the seed does)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
