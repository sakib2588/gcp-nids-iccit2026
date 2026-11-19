#!/usr/bin/env python3
"""Gap 1, cross-dataset LOO driver (plan v3 D.1). Binary, prior-insensitive.

Per held-out dataset: train on the other three (val carved from the union for
val-only selection), evaluate ONCE on the held-out dataset. Reports macro-F1,
balanced-acc, AUC, train->test delta, and a leak-free calibrated F1 (threshold
fit on a 5% target slice that is EXCLUDED from the evaluation slice).

RESUME INSTRUCTIONS:
    This script is resumable by design. If you stop mid-run (e.g., after 25/60 cells),
    simply re-run `python scripts/run_gap1.py`. It will automatically skip any cell whose
    result JSON already exists in results/gap1/*.json and only process remaining cells.

    To verify progress: ls results/gap1/*.json | grep -v gap1_summary | wc -l  # count completed cells

    The tag format is: LOO__test-{TEST_DATASET}__{HEAD}__seed{SEED}
"""
import os
import sys
import json
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_xdataset import load_loo, split_train_val_disjoint  # noqa: E402
from src.models import NIDSModel                      # noqa: E402
from src.engine import set_seed, _device             # noqa: E402
from src.metrics import macro_f1                     # noqa: E402
from src.calibration import calibrate_threshold      # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402

ALL = ["NF-UNSW-NB15-v2", "NF-CSE-CIC-IDS2018-v2", "NF-ToN-IoT-v2",
       "NF-BoT-IoT-v2"]
HEADS = ["gap", "se", "gcp"]
# Selectable but NOT default. The published 60-cell sweep is gap/se/gcp and its
# resume files are keyed on those tags, so the default stays exactly what it was.
# gmp is here because the max-pooling arm is in-distribution only, and the first
# question a reviewer asks is whether it also tracks GCP out of distribution.
ALL_HEADS = ["gap", "se", "gcp", "gmp", "gmp_wide"]
SEEDS = [42, 123, 456, 789, 1337]


def _predict(model, X, dev, batch=4096):
    """Batched forward (avoid one giant tensor on a held-out dataset -> OOM)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch].astype(np.float32)).to(dev)
            outs.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def run_one_cell(train, test, head, seed, epochs=20, batch_size=512,
                 lr=1e-3, max_rows=None, val_frac=0.15, preds_path=None,
                 parquet=None, legacy_protocol=False):
    set_seed(seed)
    dev = _device()
    # legacy_protocol reproduces the protocol that produced the published
    # gap/se/gcp cells in results/gap1/: no held-out dedup and a uniform
    # (not content-disjoint) validation carve. Reviewer run 2026-11-18, R1.2:
    # the max-pooling arm must use the same protocol as the other heads.
    Xtr_all, ytr_all, Xte, yte, meta = load_loo(
        train, test, max_rows=max_rows, parquet=parquet,
        drop_test_dupes=not legacy_protocol,
        lean=legacy_protocol)
    rng = np.random.default_rng(seed)
    # Content-disjoint carve: the union's within-corpus duplicate rate runs
    # 0.36 to 0.91, so a uniform carve selects checkpoints on rows the model
    # already trained on (audit B1).
    if legacy_protocol:
        perm = rng.permutation(len(Xtr_all))
        n_val = max(1, int(val_frac * len(Xtr_all)))
        va, tr = perm[:n_val], perm[n_val:]
        split_stats = {"n_val_dupes_dropped": 0, "val_dupe_frac": 0.0,
                       "degenerate": False}
    else:
        tr, va, split_stats = split_train_val_disjoint(
            Xtr_all, val_frac=val_frac, seed=seed)
    Xtr, ytr, Xva, yva = Xtr_all[tr], ytr_all[tr], Xtr_all[va], ytr_all[va]

    model = NIDSModel(head, 2, in_features=Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr.astype(np.float32, copy=False)),
                                  torch.from_numpy(ytr.astype(np.int64))),
                    batch_size=batch_size, shuffle=True, num_workers=0,
                    drop_last=(len(Xtr) % batch_size == 1))
    best_val, best_state = -1.0, None
    t0 = time.time()
    print(f"[gap1] {test} {head} seed={seed} train={len(Xtr)} val={len(Xva)} "
          f"test={len(Xte)} feats={Xtr.shape[1]} batches/epoch={len(dl)} "
          f"device={dev}", flush=True)
    for ep in range(epochs):
        model.train()
        for bi, (xb, yb) in enumerate(dl):
            if ep == 0 and bi == 200:
                print(f"[gap1] 200 batches in {time.time() - t0:.1f}s -> "
                      f"epoch ETA {(time.time() - t0) / 200 * len(dl) / 60:.1f}"
                      f" min", flush=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb.to(dev)), yb.to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        vf1 = macro_f1(yva, _predict(model, Xva, dev).argmax(1))
        print(f"[gap1] {test} {head} seed={seed} epoch {ep + 1}/{epochs} "
              f"val_macro_f1={vf1:.4f} elapsed={(time.time() - t0) / 60:.1f}min",
              flush=True)
        if vf1 > best_val:
            best_val = vf1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    # train F1 (generalisation gap) on a capped subsample
    tr_idx = tr[:min(len(tr), 50_000)]
    train_f1 = macro_f1(ytr_all[tr_idx],
                        _predict(model, Xtr_all[tr_idx], dev).argmax(1))

    # eval on held-out dataset (touched once)
    prob = _predict(model, Xte, dev)
    pred = prob.argmax(1)
    p_attack = prob[:, 1]
    n_cal = max(1, int(0.05 * len(yte)))
    cal_idx = rng.choice(len(yte), size=n_cal, replace=False)
    eval_mask = np.ones(len(yte), dtype=bool)
    eval_mask[cal_idx] = False
    thr, _f1c, _f105 = calibrate_threshold(yte[cal_idx], p_attack[cal_idx])
    ye, pe, pae = yte[eval_mask], pred[eval_mask], p_attack[eval_mask]
    pred_cal = (pae >= thr).astype(int)
    auc = float(roc_auc_score(ye, pae)) if len(np.unique(ye)) > 1 else None
    test_f1 = macro_f1(ye, pe)

    # Persist the raw held-out predictions. Without these the calibrated-F1
    # threshold search cannot be recomputed from disk, because the decision
    # threshold grid is the only thing that changes and the model outputs are
    # otherwise discarded. Audit item 1.12: 33 of 60 published cells selected a
    # threshold pinned to a grid endpoint, so f1_calibrated is a lower bound for
    # more than half the table and cannot be corrected without a rerun.
    # Written atomically so a crash mid-dump cannot leave a half-written array.
    if preds_path is not None:
        os.makedirs(os.path.dirname(preds_path) or ".", exist_ok=True)
        tmp_preds = preds_path + ".tmp.npz"
        np.savez_compressed(tmp_preds,
                            p_attack=p_attack.astype(np.float32),
                            y_true=yte.astype(np.int8),
                            cal_idx=cal_idx.astype(np.int32),
                            threshold=np.float32(thr))
        os.replace(tmp_preds, preds_path)

    return {"train": train, "test": test, "head": head, "seed": seed,
            "macro_f1": test_f1,
            "balanced_accuracy": float(balanced_accuracy_score(ye, pe)),
            "auc": auc, "f1_uncalibrated": test_f1,
            "f1_calibrated": float(macro_f1(ye, pred_cal)),
            "train_macro_f1": float(train_f1),
            "train_test_delta": float(train_f1 - test_f1),
            "threshold": thr, "best_val_macro_f1": float(best_val),
            "n_train": int(len(tr)), "n_val": int(len(va)),
            "n_test_eval": int(eval_mask.sum()),
            # B1 leak-check evidence, per cell, so the table can be audited
            # without re-reading the loader.
            "dedup": meta["dedup"],
            "n_test_raw": meta["n_test_raw"],
            "n_test_dupes_dropped": meta["n_test_dupes_dropped"],
            "test_dupe_frac": meta["test_dupe_frac"],
            "n_val_dupes_dropped": split_stats["n_val_dupes_dropped"],
            "val_dupe_frac": split_stats["val_dupe_frac"],
            "val_split_degenerate": split_stats["degenerate"]}


def run_gap1(results_dir="results/gap1", seeds=SEEDS, heads=HEADS,
             epochs=20, max_rows=None, save_preds=True, preds_dir=None,
             test_datasets=None, parquet=None, legacy_protocol=False):
    """Run the LOO sweep. `test_datasets` restricts which held-out corpora this
    process owns, so two hosts can split the 60 cells without racing: the
    training union is always the other three, so no comparison crosses hosts.
    Never partition by seed, that would mix GPU models into the n=5 intervals.
    """
    os.makedirs(results_dir, exist_ok=True)
    if preds_dir is None:
        preds_dir = os.path.join(results_dir, "preds")
    rows = []
    for test in (list(test_datasets) if test_datasets else ALL):
        train = [d for d in ALL if d != test]
        for head in heads:
            for seed in seeds:
                tag = f"LOO__test-{test}__{head}__seed{seed}"
                rp = os.path.join(results_dir, f"{tag}.json")
                # RESUME CHECK: skip if result already exists (loaded from disk).
                # A truncated JSON, which the pre-2026-07-20 non-atomic writer
                # could leave behind on a crash, is treated as incomplete and
                # recomputed rather than aborting the entire sweep.
                if os.path.exists(rp):
                    try:
                        with open(rp) as fh:
                            rows.append(json.load(fh))
                        continue  # goes to next cell automatically
                    except (json.JSONDecodeError, OSError) as exc:
                        print(f"[gap1] unreadable result {rp} ({exc}), "
                              f"recomputing this cell", flush=True)
                res = run_one_cell(
                    train, test, head, seed, epochs=epochs, max_rows=max_rows,
                    parquet=parquet, legacy_protocol=legacy_protocol,
                    preds_path=(os.path.join(preds_dir, f"{tag}.npz")
                                if save_preds else None))
                # Atomic write. A kernel panic partway through json.dump must
                # not leave a half-written file that the resume check trusts.
                tmp = rp + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(res, fh, indent=2)
                os.replace(tmp, rp)
                rows.append(res)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Gap 1 cross-dataset leave-one-out sweep")
    ap.add_argument("--test-datasets", nargs="+", default=ALL, choices=ALL,
                    metavar="DATASET",
                    help="held-out corpora this host owns (default: all four)")
    ap.add_argument("--results-dir", default="results/gap1")
    ap.add_argument("--parquet", default=None,
                    help="path to NF-UQ-NIDS-v2.parquet on this host "
                         "(default: src.data.DEFAULT_PARQUET)")
    ap.add_argument("--max-rows", type=int, default=2_000_000,
                    help="per-dataset row cap, a parquet-order prefix")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--heads", nargs="+", default=HEADS, choices=ALL_HEADS,
                    metavar="HEAD")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                    metavar="SEED")
    ap.add_argument("--no-preds", action="store_true",
                    help="skip the per-cell held-out prediction dump")
    ap.add_argument("--legacy-protocol", action="store_true",
                    help="no held-out dedup, uniform val carve: the protocol "
                         "of the published gap/se/gcp cells")
    args = ap.parse_args(argv)
    return run_gap1(results_dir=args.results_dir, seeds=args.seeds,
                    heads=args.heads, epochs=args.epochs,
                    max_rows=args.max_rows, save_preds=not args.no_preds,
                    test_datasets=args.test_datasets, parquet=args.parquet,
                    legacy_protocol=args.legacy_protocol)


if __name__ == "__main__":
    # 2.0M row cap is a parquet-order prefix, not a stratified sample
    # (audit 3.12): the class marginals happen to be unbiased in the current
    # file layout, but nothing in the code enforces that.
    main()
