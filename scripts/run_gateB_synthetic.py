"""T6 -- GATE B: GAP vs GCP on synthetic mean-signal vs covariance-signal data.

Theory under test (paper_writing/theory_gcp_failure.md): GCP should beat GAP
only when class separation lives in feature-pair covariance, not in simple
per-feature means. Sweeps signal in {mean, cov} x strength in [0, 1] x
head in {gap, gcp}, n=5 seeds per cell, and writes one JSON per cell to
results/gap6/.

GATE B PASS criterion: for signal="cov", GCP macro-F1 advantage over GAP grows
with strength (positive trend, delta > 0 at strength=1.0); for signal="mean",
GCP shows no such advantage (delta approx 0 or negative at strength=1.0).
"""
from __future__ import annotations

import os
import argparse
import sys
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# `python scripts/run_gateB_synthetic.py` puts scripts/ on sys.path, not the
# repo root, so the src imports below need the root added first. Every other
# driver in this directory already does this; this one did not, and the
# 2026-07-21 overnight chain died here instantly with ModuleNotFoundError.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.synthetic import generate_synthetic       # noqa: E402
from src.models import NIDSModel                   # noqa: E402
from src.metrics import macro_f1                   # noqa: E402

SEEDS = (42, 123, 456, 789, 1337)
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
SIGNALS = ("mean", "cov")
ALL_SIGNALS = ("mean", "cov", "cov_far")
HEADS = ("gap", "gcp")
# Selectable via --heads so the max-pooling arm can be added to an existing
# directory without re-running the published cells. Default unchanged.
ALL_HEADS = ("gap", "gcp", "gmp", "gmp_wide", "se", "gap_wide")
N_FEATURES = 39
N_SAMPLES = 4000
# Audit 0.2: these were 15 and 256 while Methods declared the real protocol of
# 20 and 512, so the synthetic control did not run the protocol it stood in for.
# Output goes to a v2 directory: results/gateB_synthetic/ holds the old cells
# and stays readable for the before/after comparison.
EPOCHS = 20
BATCH_SIZE = 512
OUT_DIR = "results/gateB_synthetic_v2"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _split(X, y, seed, train_frac=0.7, val_frac=0.15):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_tr = int(len(X) * train_frac)
    n_va = int(len(X) * val_frac)
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X.astype(np.float32)),
                       torch.from_numpy(y.astype(np.int64)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_cell(signal: str, strength: float, head: str, seed: int,
             n_samples: int = N_SAMPLES) -> dict:
    set_seed(seed)
    device = _device()
    X, y = generate_synthetic(n_samples, N_FEATURES, signal, strength, seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _split(X, y, seed)

    model = NIDSModel(head, num_classes=2, in_features=N_FEATURES,
                      channels=(16, 32, 32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    tr_loader = _loader(Xtr, ytr, BATCH_SIZE, shuffle=True)
    va_loader = _loader(Xva, yva, BATCH_SIZE, shuffle=False)
    te_loader = _loader(Xte, yte, BATCH_SIZE, shuffle=False)

    best_val, best_state = -1.0, None
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                p = model(xb.to(device)).argmax(1).cpu().numpy()
                preds.append(p); gts.append(yb.numpy())
        vf1 = macro_f1(np.concatenate(gts), np.concatenate(preds))
        if vf1 > best_val:
            best_val = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            p = model(xb.to(device)).argmax(1).cpu().numpy()
            preds.append(p); gts.append(yb.numpy())
    test_f1 = macro_f1(np.concatenate(gts), np.concatenate(preds))

    return {
        "signal": signal, "strength": strength, "head": head, "seed": seed,
        "val_macro_f1": best_val, "test_macro_f1": test_f1,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="GATE B synthetic covariance-vs-mean causal test")
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES,
                    help="samples per cell before the 70/15/15 split. The "
                         "default 4000 leaves only 2800 training rows to "
                         "estimate a covariance-only signal across 39 "
                         "features, which is a power concern, not a design "
                         "choice. Raise it to test whether a null at the "
                         "default is insensitivity or insufficient data.")
    ap.add_argument("--heads", nargs="+", default=list(HEADS),
                    choices=list(ALL_HEADS), metavar="HEAD",
                    help="pooling heads to run (default: gap gcp, unchanged). "
                         "Pass a single head to add an arm to a directory that "
                         "already holds the others. Only pair a new arm with "
                         "cells written by the CURRENT script: the cells in "
                         "results/gateB_synthetic_v2 carry no n_samples key, "
                         "so they predate this version and their protocol is "
                         "not established. results/gateB_synthetic_n40k does "
                         "carry it.")
    ap.add_argument("--signals", nargs="+", default=list(SIGNALS),
                    choices=list(ALL_SIGNALS), metavar="SIGNAL",
                    help="regimes to run (default: mean cov, unchanged)")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="results directory; use a fresh one per n-samples "
                         "so runs at different N never share a resume check")
    args = ap.parse_args(argv)

    out_dir = args.out_dir
    heads = list(args.heads)
    os.makedirs(out_dir, exist_ok=True)
    total = len(args.signals) * len(STRENGTHS) * len(heads) * len(SEEDS)
    done = 0
    print(f"[gateB] n_samples={args.n_samples} epochs={EPOCHS} "
          f"batch={BATCH_SIZE} heads={heads} out={out_dir}", flush=True)
    for signal in args.signals:
        for strength in STRENGTHS:
            for head in heads:
                for seed in SEEDS:
                    tag = f"{signal}__s{strength}__{head}__seed{seed}"
                    out_path = os.path.join(out_dir, f"{tag}.json")
                    if os.path.exists(out_path):
                        done += 1
                        continue
                    res = run_cell(signal, strength, head, seed,
                                   n_samples=args.n_samples)
                    res["n_samples"] = args.n_samples
                    with open(out_path, "w") as fh:
                        json.dump(res, fh, indent=2)
                    done += 1
                    print(f"[{done}/{total}] {tag} test_f1={res['test_macro_f1']:.4f}",
                         flush=True)


if __name__ == "__main__":
    main()
