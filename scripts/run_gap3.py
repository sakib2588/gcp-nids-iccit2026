#!/usr/bin/env python3
"""Gap 3 — mechanism / SHAP + H5 control (plan v3 D.3).

Records the inference-time H5 contrasts: GCP-layer covariance of top-SHAP pairs
vs (1) random pairs in the GCP model and (2) the same pairs in a GAP model's
feature map. Baseline (3) label-shuffled requires a separately-trained model and
is flagged as a deferred GPU run (the mechanism verdict is PROVISIONAL until then).
"""
import os
import sys
import json
import math

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import data as D                              # noqa: E402
from src.models import NIDSModel                       # noqa: E402
from src.engine import _device                        # noqa: E402
from src.xai.shap_runner import top_channel_pairs, layer_covariance  # noqa: E402
from src.metrics import bootstrap_ci, cohen_dz  # noqa: E402


def _pair_cov_mean(cov, pairs):
    # cov is a numpy (C,C) matrix from layer_covariance(...).cpu().numpy()
    return (float(np.mean([abs(float(cov[i, j])) for i, j in pairs]))
            if pairs else 0.0)


def _load(cp, dev):
    ck = torch.load(cp, map_location="cpu", weights_only=False)
    m = NIDSModel(ck["head"], ck["n_classes"], in_features=ck["in_features"],
                  channels=(16, 32, ck.get("c_last", 32)),
                  backbone=ck.get("backbone", "cnn"))
    m.load_state_dict(ck["model_state_dict"])
    return m.to(dev)


def _concentration(cov, pairs):
    """Mean |covariance| over the given channel pairs (top or random)."""
    return _pair_cov_mean(cov, pairs)


def _h5_verdict(real_ratios, shuffled_ratios):
    """H5 supported iff real covariance concentration exceeds the label-shuffled
    control: paired delta CI excludes zero (GCP side) AND Cohen d_z >= 0.8.
    """
    real = list(map(float, real_ratios))
    shuf = list(map(float, shuffled_ratios))
    delta = [r - s for r, s in zip(real, shuf)]
    _, lo, hi = bootstrap_ci(delta)
    dz = cohen_dz(real, shuf)
    supported = (lo > 0) and (dz >= 0.8)
    # cohen_dz is +/-inf when every seed shows the identical gain (maximally
    # consistent effect). The gate above compares correctly against inf; only
    # the serialised value needs guarding, since JSON has no infinity literal.
    finite_dz = math.isfinite(dz)
    return {"real_mean_ci": list(bootstrap_ci(real)),
            "shuffled_mean_ci": list(bootstrap_ci(shuf)),
            "delta_ci": [float(sum(delta) / len(delta)), lo, hi],
            "cohen_dz": dz if finite_dz else ("+inf" if dz > 0 else "-inf"),
            "cohen_dz_unbounded": not finite_dz,
            "supported": bool(supported)}


def run_gap3(dataset="NF-UNSW-NB15-v2", seeds=(42, 123, 456, 789, 1337), n=500,
             real_ckpt_dir="checkpoints/repro",
             shuffled_ckpt_dir="checkpoints/gap3_shuffled",
             out_dir=None, results_dir="results/gap3"):
    os.makedirs(results_dir, exist_ok=True)
    kw = {} if out_dir is None else {"out_dir": out_dir}
    Xte, _yte = D.load_split(dataset, "test", label_mode="binary", **kw)
    X = Xte[:n].astype(np.float32)
    dev = _device()
    Xt = torch.from_numpy(X).to(dev)
    rng = np.random.default_rng(42)

    per_seed = []
    real_ratio, shuffled_ratio = [], []
    for s in seeds:
        gcp = _load(os.path.join(real_ckpt_dir,
                    f"{dataset}__gcp__binary__seed{s}.pt"), dev)
        gap = _load(os.path.join(real_ckpt_dir,
                    f"{dataset}__gap__binary__seed{s}.pt"), dev)
        shuf = _load(os.path.join(shuffled_ckpt_dir,
                     f"{dataset}__gcp__binary__seed{s}__shuffled.pt"), dev)

        # channel pairs selected on the REAL gcp model, in channel space:
        # all mutual pairs among the top-5 attributed channels (10 pairs)
        pairs = top_channel_pairs(gcp, Xt, n_channels=5)
        cov_gcp = layer_covariance(gcp, Xt).cpu().numpy()
        cov_gap = layer_covariance(gap, Xt).cpu().numpy()
        cov_shuf = layer_covariance(shuf, Xt).cpu().numpy()
        C = cov_gcp.shape[0]
        # matched-count random OFF-DIAGONAL pairs (a != b), so the baseline is
        # apples-to-apples with the off-diagonal top pairs (no self/variance terms)
        rand_pairs = []
        while len(rand_pairs) < len(pairs):
            a, b = int(rng.integers(0, C)), int(rng.integers(0, C))
            if a != b:
                rand_pairs.append((a, b))

        top_gcp = _concentration(cov_gcp, pairs)
        rnd_gcp = _concentration(cov_gcp, rand_pairs)
        top_gap = _concentration(cov_gap, pairs)
        top_shuf = _concentration(cov_shuf, pairs)
        rnd_shuf = _concentration(cov_shuf, rand_pairs)
        # concentration ratio = top-pair cov / random-pair cov, WITHIN each model
        # (r_shuf uses the shuffled model's OWN baseline, not the real model's)
        r_real = top_gcp / rnd_gcp if rnd_gcp else 0.0
        r_shuf = top_shuf / rnd_shuf if rnd_shuf else 0.0
        real_ratio.append(r_real)
        shuffled_ratio.append(r_shuf)
        per_seed.append({"seed": s, "top_pair_cov_gcp": top_gcp,
                         "random_pair_cov_gcp": rnd_gcp,
                         "top_pair_cov_gap_featuremap": top_gap,
                         "top_pair_cov_gcp_shuffled": top_shuf,
                         "random_pair_cov_gcp_shuffled": rnd_shuf,
                         "concentration_real": r_real,
                         "concentration_shuffled": r_shuf})

    verdict = _h5_verdict(real_ratio, shuffled_ratio)
    out = {"dataset": dataset, "seeds": list(seeds), "n_eval": n,
           "attribution": "gradient-x-activation channel attribution (Grad-CAM "
                          "style) in GCP-layer channel space; NOT input-feature SHAP",
           "pair_definition": "top pairs = all C(5,2)=10 mutual pairs among the "
                              "top-5 attributed channels; random baseline = 10 "
                              "off-diagonal (a!=b) random channel pairs; "
                              "concentration ratio computed WITHIN each model",
           "per_seed": per_seed, "h5": verdict,
           "note": "H5 label-shuffled control INCLUDED; verdict is final for "
                   "this dataset. Single-dataset (UNSW) mechanism illustration."}
    with open(os.path.join(results_dir, "gap3_summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    return out


if __name__ == "__main__":
    print(run_gap3())
