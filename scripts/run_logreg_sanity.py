#!/usr/bin/env python3
"""Logistic-regression sanity check on the synthetic generators (reviewer R1.3).

Reviewer 1.3 asked that, once the synthetic generator matches per-feature mean
and variance across classes, a logistic regression on means and variances
scores near chance. This script runs that check on the exact data the head
comparison used: src.synthetic.generate_synthetic, N=40,000, 39 features, the
same seeds and the same 70/15/15 permutation split as
scripts/run_gateB_synthetic.py::_split.

Two probes, both fitted on the training split and scored once on the test split
(macro-F1, the paper's metric):
  first_order  features x and x^2 (per-feature mean and variance shifts are
               linearly visible here; no cross-feature product is available)
  products     x_i * x_j for all i < j (741 features), a second-order probe that
               shows the covariance signal exists in the data

Expected: first_order is near 0.5 on cov_far at every strength and clearly
above chance on the mean regime; products rises with strength on cov_far.

Usage (from the repo root):
  uv run --with numpy --with scikit-learn python scripts/run_logreg_sanity.py
Writes results/gateB_synthetic_far/logreg_sanity.json.
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics as st
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load src/synthetic.py by path: src/__init__ pulls in torch, which this check
# does not need.
_spec = importlib.util.spec_from_file_location(
    "synthetic", os.path.join(ROOT, "src", "synthetic.py"))
_syn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_syn)
generate_synthetic = _syn.generate_synthetic

SEEDS = (42, 123, 456, 789, 1337)
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
REGIMES = ("cov_far", "mean", "cov")
N_SAMPLES, N_FEATURES = 40000, 39
OUT = os.path.join(ROOT, "results", "gateB_synthetic_far", "logreg_sanity.json")


def split(X, y, seed, train_frac=0.7, val_frac=0.15):
    """Identical to scripts/run_gateB_synthetic.py::_split."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_tr = int(len(X) * train_frac)
    n_va = int(len(X) * val_frac)
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def first_order(X):
    return np.hstack([X, X ** 2])


def products(X):
    iu = np.triu_indices(X.shape[1], k=1)
    return X[:, iu[0]] * X[:, iu[1]]


def fit_score(F, y, tr, te):
    sc = StandardScaler().fit(F[tr])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(sc.transform(F[tr]), y[tr])
    pred = clf.predict(sc.transform(F[te]))
    return float(f1_score(y[te], pred, average="macro"))


def main():
    out = {"_meta": {"n_samples": N_SAMPLES, "n_features": N_FEATURES,
                     "seeds": list(SEEDS), "strengths": list(STRENGTHS),
                     "model": "sklearn LogisticRegression, L2 C=1.0, standardized",
                     "metric": "macro-F1 on the 15 percent test split",
                     "split": "scripts/run_gateB_synthetic.py::_split"},
           "cells": {}}
    for regime in REGIMES:
        for s in STRENGTHS:
            row = {"first_order": [], "products": []}
            for seed in SEEDS:
                X, y = generate_synthetic(N_SAMPLES, N_FEATURES, regime, s, seed)
                tr, _, te = split(X, y, seed)
                row["first_order"].append(fit_score(first_order(X), y, tr, te))
                row["products"].append(fit_score(products(X), y, tr, te))
            out["cells"][f"{regime}__s{s}"] = row
            fo, pr = row["first_order"], row["products"]
            print(f"{regime:8s} s={s:4.2f}  first_order {st.mean(fo):.4f} "
                  f"[{min(fo):.4f},{max(fo):.4f}]   products {st.mean(pr):.4f} "
                  f"[{min(pr):.4f},{max(pr):.4f}]", flush=True)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", OUT_PATH)


OUT_PATH = os.environ.get("LOGREG_OUT", OUT)

if __name__ == "__main__":
    main()
