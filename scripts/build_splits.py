#!/usr/bin/env python3
"""Build the four NetFlow-v2 train/validation/test splits used in the paper.

Reads the combined NF-UQ-NIDS-v2 parquet (see README.md for the download and the
CSV-to-parquet step), keeps each source dataset by its ``Dataset`` column, and calls
``src.data.materialize_dataset`` with the paper's settings:

  * seed 42, stratified 70/15/15 split on the native multiclass label
  * leakage-only deduplication: validation and test rows whose feature hash occurs
    in training are dropped, training rows are kept
  * constant columns dropped and z-scoring fit on the training split only
  * the three larger corpora capped at 400,000 rows per class; NF-UNSW-NB15-v2 uncapped

It prints the row counts next to the ones the paper used, so a rebuilt split can be
checked against the original. Usage:

    python scripts/build_splits.py                    # all four datasets
    python scripts/build_splits.py --datasets NF-BoT-IoT-v2
    NF_UQ_PARQUET=/path/to/NF-UQ-NIDS-v2.parquet python scripts/build_splits.py
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import DATASETS, DEFAULT_OUT, DEFAULT_PARQUET, materialize_dataset  # noqa: E402

CAP = {"NF-UNSW-NB15-v2": None, "NF-CSE-CIC-IDS2018-v2": 400_000,
       "NF-ToN-IoT-v2": 400_000, "NF-BoT-IoT-v2": 400_000}

# Row counts of the splits behind every reported number (meta.json of the original build).
EXPECTED = {
    "NF-UNSW-NB15-v2":       dict(n_total=2390275, n_train=1673191, n_val=23214, n_test=20844, n_features=39),
    "NF-CSE-CIC-IDS2018-v2": dict(n_total=1583872, n_train=1108710, n_val=92016, n_test=90636, n_features=38),
    "NF-ToN-IoT-v2":         dict(n_total=2827957, n_train=1979569, n_val=94022, n_test=89415, n_features=39),
    "NF-BoT-IoT-v2":         dict(n_total=1337468, n_train=936226, n_val=5020, n_test=4525, n_features=39),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--parquet", default=DEFAULT_PARQUET)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()
    if not os.path.exists(args.parquet):
        sys.exit(f"parquet not found: {args.parquet}\nSee README.md, section Data.")
    ok = True
    for name in args.datasets:
        meta = materialize_dataset(name, parquet=args.parquet, out_dir=args.out_dir,
                                   seed=42, per_class_cap=CAP[name])
        for k, want in EXPECTED[name].items():
            got = meta.get(k)
            flag = "ok" if got == want else "MISMATCH"
            ok &= got == want
            print(f"{name:24s} {k:10s} {got!s:>9} (paper {want}) {flag}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
