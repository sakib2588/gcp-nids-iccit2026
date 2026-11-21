#!/usr/bin/env python3
"""Audit item B12, direct test: rerun Extra Trees on the BoT-IoT split for several seeds.

Read-only on the split directory. Uses the exact classifier call of
scripts/run_extratrees_baseline.py (ExtraTreesClassifier, 100 trees, random_state=seed)
and the repository's macro_f1. Prints the new value next to the stored one and the
number of test rows whose predicted label differs between seeds.

Usage: python3 scripts/rerun_extratrees_bot_check.py SPLIT_DIR [seed ...]
"""
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import macro_f1  # noqa: E402

base = sys.argv[1]
seeds = [int(s) for s in sys.argv[2:]] or [42, 123]
meta = json.load(open(os.path.join(base, "meta.json")))
print("meta keys:", [k for k in meta.keys()][:12])
Xtr, ytr = np.load(f"{base}/train_X.npy"), np.load(f"{base}/train_ybin.npy")
Xte, yte = np.load(f"{base}/test_X.npy"), np.load(f"{base}/test_ybin.npy")
print("train", Xtr.shape, "test", Xte.shape, "test attack share", round(float(yte.mean()), 4))
print("test_X sha256[:16]", hashlib.sha256(np.ascontiguousarray(Xte).tobytes()).hexdigest()[:16])
preds = {}
for s in seeds:
    clf = ExtraTreesClassifier(n_estimators=100, random_state=s, n_jobs=4)
    clf.fit(Xtr, ytr)
    p = clf.predict(Xte)
    preds[s] = p
    stored = json.load(open(f"results/extratrees/NF-BoT-IoT-v2__extratrees__binary__seed{s}.json"))["macro_f1"]
    print(f"seed {s:5d}  rerun macro_f1={macro_f1(yte, p):.6f}  stored={stored:.6f}  errors={int((p != yte).sum())}")
ss = sorted(preds)
for i in range(len(ss)):
    for j in range(i + 1, len(ss)):
        print(f"seeds {ss[i]} vs {ss[j]}: predicted labels differ on {int((preds[ss[i]] != preds[ss[j]]).sum())} of {len(yte)} test rows")
