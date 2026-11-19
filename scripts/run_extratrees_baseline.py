#!/usr/bin/env python3
"""Extra Trees under OUR protocol, as a matched published-method baseline.

Sarhan et al. report Extra Trees on exactly these four NF-v2 corpora, but under a
different protocol (random 70/30, no leakage control, natural class balance,
attack-class binary F1). Quoting their numbers therefore places our backbone only
loosely. This runs the same classifier family on OUR splits, with our leakage
control, our per-class cap and our macro-F1, so the comparison is matched.

Identical split path to scripts/run_tree_baseline.py, which is the HGB baseline
already in the paper, so the three model families share train/val/test arrays.
"""
from __future__ import annotations

import argparse, json, os, sys, time
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import data as D                       # noqa: E402
from src.engine import _atomic_json_dump        # noqa: E402
from src.metrics import macro_f1, t_ci          # noqa: E402

DATASETS = ["NF-UNSW-NB15-v2", "NF-BoT-IoT-v2",
            "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v2"]
SEEDS = [42, 123, 456, 789, 1337]
RESULTS_DIR = "results/extratrees"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--time-one", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    total, done = len(a.datasets) * len(a.seeds), 0
    for ds in a.datasets:
        Xtr, ytr = D.load_split(ds, "train", label_mode="binary")
        Xva, yva = D.load_split(ds, "val", label_mode="binary")
        Xte, yte = D.load_split(ds, "test", label_mode="binary")
        for seed in a.seeds:
            tag = f"{ds}__extratrees__binary__seed{seed}"
            path = os.path.join(a.out_dir, f"{tag}.json")
            done += 1
            if os.path.exists(path):
                print(f"[{done}/{total}] {tag} SKIP (exists)", flush=True)
                continue
            t0 = time.time()
            clf = ExtraTreesClassifier(n_estimators=a.n_estimators,
                                       random_state=seed, n_jobs=a.n_jobs)
            clf.fit(Xtr, ytr)
            rec = {"macro_f1": float(macro_f1(yte, clf.predict(Xte))),
                   "val_macro_f1": float(macro_f1(yva, clf.predict(Xva))),
                   "dataset": ds, "head": "extratrees", "label_mode": "binary",
                   "seed": seed, "n_test": int(len(yte)),
                   "n_features": int(Xtr.shape[1]),
                   "n_estimators": a.n_estimators,
                   "fit_predict_seconds": round(time.time() - t0, 1)}
            _atomic_json_dump(rec, path)
            print(f"[{done}/{total}] {tag} macro_f1={rec['macro_f1']:.4f} "
                  f"({rec['fit_predict_seconds']/60:.1f} min)", flush=True)
            if a.time_one:
                print(f"[timing] one cell took {rec['fit_predict_seconds']/60:.1f} min; "
                      f"{total} cells projects to ~{rec['fit_predict_seconds']*total/3600:.1f} h "
                      f"(dataset sizes differ, so this is an anchor, not a total)", flush=True)
                return

    summary = {"_method": "sklearn ExtraTreesClassifier, sklearn defaults, untuned",
               "_protocol": "our splits, our leakage control, our per-class cap, macro-F1",
               "datasets": {}}
    for ds in a.datasets:
        vals = []
        for s in a.seeds:
            p = os.path.join(a.out_dir, f"{ds}__extratrees__binary__seed{s}.json")
            if not os.path.exists(p):
                break
            vals.append(json.load(open(p))["macro_f1"])
        if len(vals) == len(a.seeds):
            m, lo, hi = t_ci(vals)
            summary["datasets"][ds] = {"extratrees_macro_f1_t_ci": [m, lo, hi],
                                       "per_seed": vals,
                                       "mean": float(np.mean(vals)),
                                       "std": float(np.std(vals, ddof=1))}
    _atomic_json_dump(summary, os.path.join(a.out_dir, "extratrees_summary.json"))
    print("extratrees baseline done", flush=True)


if __name__ == "__main__":
    main()
