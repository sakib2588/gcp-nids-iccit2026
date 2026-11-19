"""R1+ -- gradient-boosted-tree baseline on all four NetFlow datasets.

WHY THIS EXISTS
The paper compares two pooling heads inside one 1D-CNN backbone and reports a
+0.0046 macro-F1 advantage. On 38-39 feature tabular NetFlow, gradient-boosted
trees are the standard reference and frequently beat 1D CNNs. Without one, the
GAP-vs-GCP difference is unanchored: it may be a comparison between two points
well below the practical frontier. This script supplies the anchor.

FAIRNESS -- the whole value of this baseline is that nothing differs but the
model family:
  - identical train/val/test arrays via src.data.load_split (no re-split, no
    re-dedup, no re-scale). z-scoring is monotone so it is inert for trees.
  - identical selection rule: the CNN picks its best EPOCH by val macro-F1, so
    the tree picks its best BOOSTING ITERATION by val macro-F1.
  - test read exactly once, at the selected iteration.
  - same five seeds, same binary task, no class weighting (CNN default).

Why not sklearn's own early stopping: HistGradientBoostingClassifier has no
external-validation parameter, and `validation_fraction` carves a random slice
out of the passed TRAINING data. That is a different protocol from the CNN's.
So early_stopping is off, the model is fit to a fixed iteration budget, and
selection runs over staged_predict on the real val split.

WHAT IS NOT COMPARABLE (recorded in the summary as _comparability):
  - interval WIDTHS. CNN seed variance comes from weight init and batch order;
    the tree is deterministic given random_state except for the binning
    subsample. "The tree is more stable" would be an artefact.
  - compute budget. 20 epochs against 500 boosting iterations, neither tuned.
  - the iteration ceiling. If val macro-F1 is still rising at the last
    iteration the tree was budget-limited, so best_iter is recorded per cell.

Resumable: skips any (dataset, seed) whose result JSON already exists.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import data as D  # noqa: E402
from src.engine import _atomic_json_dump  # noqa: E402
from src.metrics import macro_f1, t_ci  # noqa: E402


def usable_result(path):
    """True only if the cell result exists AND parses AND carries macro_f1.

    Result files are resume markers, so a truncated one that merely EXISTS
    would be skipped forever and silently lost. Writes are atomic, but verify
    anyway -- a file from an earlier tool or a full disk can still be short.
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

DATASETS = ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2",
            "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v2"]
SEEDS = [42, 123, 456, 789, 1337]
MAX_ITER = 500
RESULTS_DIR = "results/tree_baseline"

# Where each dataset's CNN comparator actually lives. NF-BoT-IoT-v2 has no
# __binary__ headline cell: its comparator is the no-resampling control of the
# Gap 4 imbalance study, same splits and protocol under a different tag.
COMPARATORS = {
    "NF-UNSW-NB15-v2": ("results/repro", "{ds}__{head}__binary__seed{seed}.json"),
    "NF-CSE-CIC-IDS2018-v2": ("results/gateC", "{ds}__{head}__binary__seed{seed}.json"),
    "NF-ToN-IoT-v2": ("results/gateC", "{ds}__{head}__binary__seed{seed}.json"),
    "NF-BoT-IoT-v2": ("results/gap4", "{ds}__{head}__none__seed{seed}.json"),
}


def select_and_score(clf, Xva, yva, Xte, yte):
    """Pick the boosting iteration by val macro-F1, then score test once there."""
    val_curve = [macro_f1(yva, p) for p in clf.staged_predict(Xva)]
    best_iter = int(np.argmax(val_curve))
    for i, pred in enumerate(clf.staged_predict(Xte)):
        if i == best_iter:
            test_pred = pred
            break
    return {
        "best_iter": best_iter + 1,
        "n_iter_fitted": len(val_curve),
        "val_macro_f1": float(val_curve[best_iter]),
        "val_macro_f1_at_last_iter": float(val_curve[-1]),
        # If the best iteration IS the last one, the budget bound the result.
        "budget_limited": bool(best_iter == len(val_curve) - 1),
        "macro_f1": float(macro_f1(yte, test_pred)),
    }


def cnn_comparators(dataset, seeds):
    """Mean +/- t interval of each CNN head's TEST macro-F1, from disk.

    Val numbers are never comparable to test numbers, so only macro_f1 is read.
    """
    out = {}
    if dataset not in COMPARATORS:
        return out
    d, pattern = COMPARATORS[dataset]
    for head in ("gap", "se", "gcp"):
        vals = []
        for s in seeds:
            p = os.path.join(d, pattern.format(ds=dataset, head=head, seed=s))
            # Published comparators are old and stable, but an unreadable one
            # must degrade to "no comparator" rather than kill a 9-hour run at
            # the summary step.
            if usable_result(p):
                with open(p) as f:
                    vals.append(json.load(f)["macro_f1"])
        if len(vals) == len(seeds):
            m, lo, hi = t_ci(vals)
            out[head] = {"macro_f1_t_ci": [m, lo, hi], "n": len(vals),
                         "source_dir": d}
    return out


def verdict(tree_ci, head_ci):
    """Overlapping Student-t intervals count as a tie, not a win.

    Guards n<2: t_ci at n=1 returns [-inf, +inf], which overlaps everything and
    would silently report "tie" for every comparison -- a real-looking verdict
    carrying no information. Seen live on a single-seed smoke run.
    """
    _tm, tlo, thi = tree_ci
    _hm, hlo, hhi = head_ci
    if not all(np.isfinite(v) for v in (tlo, thi, hlo, hhi)):
        return "undetermined_n_too_small"
    if tlo > hhi:
        return "tree_wins"
    if thi < hlo:
        return "tree_loses"
    return "tie"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--max-iter", type=int, default=MAX_ITER)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    total = len(args.datasets) * len(args.seeds)
    done = 0
    for ds in args.datasets:
        Xtr, ytr = D.load_split(ds, "train", label_mode="binary")
        Xva, yva = D.load_split(ds, "val", label_mode="binary")
        Xte, yte = D.load_split(ds, "test", label_mode="binary")
        for seed in args.seeds:
            tag = f"{ds}__hgb__binary__seed{seed}"
            path = os.path.join(args.out_dir, f"{tag}.json")
            done += 1
            if usable_result(path):
                print(f"[{done}/{total}] {tag} SKIP (exists)", flush=True)
                continue
            clf = HistGradientBoostingClassifier(
                max_iter=args.max_iter, early_stopping=False,
                random_state=seed)
            clf.fit(Xtr, ytr)
            rec = select_and_score(clf, Xva, yva, Xte, yte)
            rec.update({"dataset": ds, "head": "hgb", "label_mode": "binary",
                        "seed": seed, "n_test": int(len(yte)),
                        "n_features": int(Xtr.shape[1])})
            _atomic_json_dump(rec, path)
            print(f"[{done}/{total}] {tag} macro_f1={rec['macro_f1']:.4f} "
                  f"best_iter={rec['best_iter']}"
                  f"{' BUDGET-LIMITED' if rec['budget_limited'] else ''}",
                  flush=True)

    # ---- summary -----------------------------------------------------------
    summary = {"datasets": {}}
    for ds in args.datasets:
        vals, iters, limited = [], [], 0
        for s in args.seeds:
            p = os.path.join(args.out_dir, f"{ds}__hgb__binary__seed{s}.json")
            if not usable_result(p):
                break
            with open(p) as f:
                r = json.load(f)
            vals.append(r["macro_f1"]); iters.append(r["best_iter"])
            limited += int(r["budget_limited"])
        if len(vals) != len(args.seeds):
            continue
        m, lo, hi = t_ci(vals)
        block = {"hgb_macro_f1_t_ci": [m, lo, hi], "per_seed": vals,
                 "best_iter_per_seed": iters,
                 "n_seeds_budget_limited": limited,
                 "cnn": cnn_comparators(ds, args.seeds)}
        block["verdict_vs"] = {h: verdict([m, lo, hi], v["macro_f1_t_ci"])
                               for h, v in block["cnn"].items()}
        summary["datasets"][ds] = block

    summary["_comparability"] = {
        "interval_widths": ("NOT comparable across model families. CNN seed "
                            "variance is weight init and batch order; HGB is "
                            "deterministic given random_state except for the "
                            "binning subsample (measured SD ~9e-4). 'The tree "
                            "is more stable' would be an artefact."),
        "compute_budget": ("NOT matched. 20 CNN epochs against "
                           f"{args.max_iter} boosting iterations, neither tuned."),
        "iteration_ceiling": ("check n_seeds_budget_limited: if non-zero, val "
                              "macro-F1 was still rising at the last iteration "
                              "and the tree was budget-limited."),
        "point_estimates": "comparable -- same splits, same selection rule, same seeds.",
    }
    summary["_label_mode_coverage"] = ("binary only; results/multiclass/ has no "
                                       "tree anchor, so the 15-family analysis "
                                       "is unanchored by this baseline.")
    summary["_comparators"] = {ds: COMPARATORS.get(ds, ("", ""))[0]
                               for ds in args.datasets}
    summary["_method"] = {
        "model": "sklearn HistGradientBoostingClassifier, early_stopping=False",
        "selection": "boosting iteration argmax of val macro-F1 (mirrors the "
                     "CNN's best-epoch-by-val rule)",
        "test_policy": "read once, at the selected iteration",
        "interval": "Student-t, df = n-1",
        "verdict_rule": "overlapping Student-t intervals count as a tie",
    }

    out = os.path.join(args.out_dir, "tree_baseline_summary.json")
    _atomic_json_dump(summary, out)

    print("\nTEST macro-F1, tree against the CNN heads (Student-t, n=5):")
    for ds, b in summary["datasets"].items():
        m, lo, hi = b["hgb_macro_f1_t_ci"]
        print(f"\n  {ds}")
        print(f"    HGB  {m:.4f} [{lo:.4f}, {hi:.4f}]"
              f"{'  (budget-limited on %d/%d seeds)' % (b['n_seeds_budget_limited'], len(args.seeds)) if b['n_seeds_budget_limited'] else ''}")
        for h, v in b["cnn"].items():
            hm, hlo, hhi = v["macro_f1_t_ci"]
            print(f"    {h.upper():<4} {hm:.4f} [{hlo:.4f}, {hhi:.4f}]"
                  f"   -> {b['verdict_vs'][h]}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
