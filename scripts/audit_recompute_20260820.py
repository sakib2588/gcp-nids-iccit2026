#!/usr/bin/env python3
"""Recompute, and persist to disk, every number the 2026-08-20 audit needs.

Why this exists. Three claims in the ICCIT manuscript had no producing file under
results/, and one arm (the max-pooling / GMP arm added 2026-08-15) never received the
Student-t interval migration decided on 2026-07-21, so its published intervals are still
percentile bootstrap while Methods declares Student-t.

This script recomputes those quantities from the per-seed cell files that are already on
disk. It reads only. It writes one new directory, results/audit_20260820/, and modifies
no existing results file.

Outputs
  results/audit_20260820/audit_summary.json

Estimator. Student-t with n-1 = 4 degrees of freedom, matching the paper's declared
method. src/metrics.py measures the percentile bootstrap at 0.839 coverage against 0.946
for the t interval at n=5, which is why the project moved.
"""
import json
import math
import os
from statistics import mean, stdev

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [42, 123, 456, 789, 1337]
DATASETS = [
    "NF-UNSW-NB15-v2",
    "NF-CSE-CIC-IDS2018-v2",
    "NF-ToN-IoT-v2",
    "NF-BoT-IoT-v2",
]
# scipy is not a hard dependency of this repo, so the two t quantiles are inlined.
T_975_DF4 = 2.7764451051977987   # two-sided 95 percent
T_800_DF4 = 0.9409645656213734   # one-sided 80 percent, for the power calculation


def t_ci(values):
    """Paired Student-t 95 percent interval for the mean of `values`."""
    m = mean(values)
    sd = stdev(values)
    half = T_975_DF4 * sd / math.sqrt(len(values))
    return {
        "mean": m,
        "t_ci": [m - half, m + half],
        "excludes_zero": not (m - half < 0 < m + half),
        "d_z": m / sd if sd else float("nan"),
        "sd": sd,
        "n": len(values),
    }


def mde(values):
    """Smallest paired difference this design resolves at 80 percent power, alpha .05."""
    return (T_975_DF4 + T_800_DF4) * stdev(values) / math.sqrt(len(values))


def _read(path, key):
    with open(path) as fh:
        return json.load(fh)[key]


def loo_cells(dataset, head, metric):
    return [
        _read(
            os.path.join(ROOT, "results", "gap1",
                         f"LOO__test-{dataset}__{head}__seed{seed}.json"),
            metric,
        )
        for seed in SEEDS
    ]


def gatec_cells(dataset, head):
    return [
        _read(
            os.path.join(ROOT, "results", "gateC",
                         f"{dataset}__{head}__binary__seed{seed}.json"),
            "macro_f1",
        )
        for seed in SEEDS
    ]


def build():
    out = {
        "_description": (
            "Audit recomputation of 2026-08-20. Student-t df=4 intervals for the "
            "cross-dataset arm including the max-pooling head, the in-distribution "
            "GCP-minus-SE contrasts, minimum detectable effects, and the "
            "single-family-drop coupling correlation. Recomputed from per-seed cell "
            "files, not copied from any summary."
        ),
        "_estimator": "Student-t, n-1 = 4 degrees of freedom",
        "_generated_by": "scripts/audit_recompute_20260820.py",
        "_reads_only": True,
    }

    # 1. Cross-dataset LOO, every head including gmp, both metrics.
    loo = {}
    for dataset in DATASETS:
        block = {}
        heads = {}
        for head in ["gap", "gcp", "se", "gmp"]:
            for metric in ["macro_f1", "auc"]:
                heads.setdefault(head, {})[metric] = loo_cells(dataset, head, metric)
        block["per_head_mean"] = {
            h: {m: mean(v) for m, v in mv.items()} for h, mv in heads.items()
        }
        contrasts = {}
        for a, b in [("gcp", "gmp"), ("gmp", "gap"), ("gcp", "gap")]:
            for metric in ["macro_f1", "auc"]:
                paired = [x - y for x, y in zip(heads[a][metric], heads[b][metric])]
                contrasts[f"{a} - {b} / {metric}"] = t_ci(paired)
        block["contrasts"] = contrasts
        loo[dataset] = block
    out["cross_dataset_loo"] = loo

    # 2. In-distribution GCP minus SE. Only the corpora with gateC cells for both heads.
    se = {}
    for dataset in DATASETS:
        try:
            gcp = gatec_cells(dataset, "gcp")
            sehead = gatec_cells(dataset, "se")
        except FileNotFoundError as exc:
            se[dataset] = {"unavailable": os.path.basename(exc.filename)}
            continue
        se[dataset] = t_ci([a - b for a, b in zip(gcp, sehead)])
    out["in_distribution_gcp_minus_se"] = se

    # 3. Minimum detectable effect, from the max-pooling summary's per-seed deltas.
    with open(os.path.join(ROOT, "results", "maxpool", "maxpool_summary.json")) as fh:
        maxpool = json.load(fh)
    mdes = {}
    for dataset, block in maxpool["real"].items():
        entry = {}
        for contrast in block["contrasts"]:
            if contrast["contrast"] in ("gcp - gap", "gcp - gmp"):
                deltas = list(contrast["per_seed_delta"].values())
                entry[contrast["contrast"]] = {
                    "mde_80pct_power": mde(deltas),
                    "observed_delta": contrast["delta_mean"],
                }
        mdes[dataset] = entry
    out["minimum_detectable_effect"] = {
        "_definition": (
            "Smallest paired macro-F1 difference detectable at alpha .05 two-sided and "
            "80 percent power, given the observed per-seed spread at n=5. "
            "(t_.975,4 + t_.80,4) * sd / sqrt(n)."
        ),
        "by_dataset": mdes,
    }

    # 4. Equivalence bound. Does the GCP-GMP interval exclude the whole GCP-GAP gain?
    bound = {}
    for dataset, block in maxpool["real"].items():
        by = {c["contrast"]: c for c in block["contrasts"]}
        if "gcp - gmp" in by and "gcp - gap" in by:
            upper = by["gcp - gmp"]["delta_t_ci"][1]
            gain = by["gcp - gap"]["delta_mean"]
            bound[dataset] = {
                "gcp_minus_gmp_upper": upper,
                "gcp_minus_gap_gain": gain,
                "interval_excludes_whole_gain": upper < gain,
            }
    out["equivalence_bound"] = bound

    # 5. Coupling correlation with the smallest family dropped.
    path = os.path.join(ROOT, "results", "efc", "efc_gcp_correlation_corrected.json")
    with open(path) as fh:
        efc = json.load(fh)
    rows = efc["rows"]
    smallest = {"dataset": "NF-CSE-CIC-IDS2018-v2", "family": "injection"}
    kept = [
        r for r in rows
        if not (r["dataset"] == smallest["dataset"] and r["family"] == smallest["family"])
    ]

    def pearson(xs, ys):
        n = len(xs)
        mx, my = mean(xs), mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        return num / den if den else float("nan")

    drop = {}
    for variant in ["index_published", "index_block_normalised",
                    "index_corrected", "index_corrected_unnormalised"]:
        drop[variant] = {
            "pearson_r_all15": pearson([r[variant] for r in rows],
                                       [r["gcp_gain"] for r in rows]),
            "pearson_r_drop_smallest": pearson([r[variant] for r in kept],
                                               [r["gcp_gain"] for r in kept]),
        }
    out["coupling_drop_smallest_family"] = {
        "_why": (
            "The manuscript states that dropping the one family whose gain rests on five "
            "test flows moves the headline coefficient and reverses the one positive "
            "variant. That claim previously had no producing file."
        ),
        "dropped": smallest,
        "n_families_after_drop": len(kept),
        "variants": drop,
    }

    # 6. Held-out AUC, for the transfer sentence.
    cells = []
    for dataset in DATASETS:
        for head in ["gap", "gcp", "gmp"]:
            cells.append(mean(loo_cells(dataset, head, "auc")))
    out["held_out_auc"] = {
        "n_cells": len(cells),
        "at_or_below_chance": sum(1 for c in cells if c <= 0.5),
        "below_0_60": sum(1 for c in cells if c < 0.60),
        "by_dataset_head": {
            dataset: {head: mean(loo_cells(dataset, head, "auc"))
                      for head in ["gap", "gcp", "gmp"]}
            for dataset in DATASETS
        },
    }
    return out


def main():
    target_dir = os.path.join(ROOT, "results", "audit_20260820")
    os.makedirs(target_dir, exist_ok=True)
    target = os.path.join(target_dir, "audit_summary.json")
    with open(target, "w") as fh:
        json.dump(build(), fh, indent=2)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
