"""Z3 -- the synthetic covariance-vs-mean regime contrast, done properly.

WHY THIS EXISTS
The manuscript calls the regime pattern "a direct causal confirmation" while
resting it on two rank correlations that OVERLAP: covariance rho=+0.455
[0.189,0.728] against mean rho=+0.310 [-0.040,0.591]. The bootstrapped
difference of those rhos crosses zero, so the stated argument does not support
the stated claim and has to be replaced.

This script computes the replacement and writes it to disk, so every number
the paper quotes traces to a file.

WHAT IT REPORTS, AND THE ORDER IS DELIBERATE
The per-strength table comes FIRST, before the pooled figure, because the
pooled contrast is resolved only at strengths 0.25 and 0.5 and COVERS ZERO at
0.75 and 1.0 -- the two endpoints the mechanism predicts should be strongest.
A reader must not be able to lift the pooled number without meeting that.

Estimators, and why:
  - Student-t df=n-1, never percentile bootstrap. At n=5 the bootstrap of a
    mean has 0.839 coverage (src/metrics.py, notes/20260720-interval-comparison).
  - Unpaired Welch is the headline interval. Pairing shrinks the SE by ~3
    percent (across-regime seed correlation is only ~+0.065) but costs df
    7.4 -> 4, so the paired interval is WIDER, and the paired sign-flip test is
    pinned at the n=5 floor of 2/32 = 0.0625 and can never clear 0.05.
  - Exact permutations, fully enumerated: 2^5 = 32 sign-flips (paired) and
    C(10,5) = 252 regime-label assignments (unpaired). No sampling.
  - The label permutation tests a STRONGER null than the sign-flip: it assumes
    a seed's cov and mean values are exchangeable across the group boundary,
    not merely that the regime label is exchangeable within a seed. Cheap at
    r=+0.065, but it is not free and the paper should not pretend it is.

Analysis-only: no GPU, no training, reads per-seed cell JSONs.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import t_ci, cohen_dz  # noqa: E402

RUNS = {"N40000": "results/gateB_synthetic_n40k",
        "N4000": "results/gateB_synthetic_v2"}
STRENGTHS = [0.0, 0.25, 0.5, 0.75, 1.0]
NONZERO = [0.25, 0.5, 0.75, 1.0]
SEEDS = [42, 123, 456, 789, 1337]


# ---------- data ------------------------------------------------------------

def cell_path(run_dir, regime, strength, head, seed):
    return os.path.join(run_dir, f"{regime}__s{strength}__{head}__seed{seed}.json")


def load_f1(run_dir, regime, strength, head, seed):
    with open(cell_path(run_dir, regime, strength, head, seed)) as f:
        return json.load(f)["test_macro_f1"]


def deltas(run_dir, regime, strength, seeds=SEEDS):
    """Paired GCP-GAP delta, one per seed, at a fixed (regime, strength)."""
    return np.array([load_f1(run_dir, regime, strength, "gcp", s)
                     - load_f1(run_dir, regime, strength, "gap", s)
                     for s in seeds])


def pooled_by_seed(run_dir, regime, strengths, seeds=SEEDS):
    """Per seed, the delta averaged over the given strengths. One value/seed."""
    return np.array([np.mean([load_f1(run_dir, regime, st, "gcp", s)
                              - load_f1(run_dir, regime, st, "gap", s)
                              for st in strengths]) for s in seeds])


# ---------- estimators ------------------------------------------------------

def welch(a, b):
    """Unpaired Welch interval for mean(a)-mean(b). Returns (diff, lo, hi, df)."""
    from scipy import stats
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    diff = a.mean() - b.mean()
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    crit = stats.t.ppf(0.975, df)
    return float(diff), float(diff - crit * se), float(diff + crit * se), float(df)


def exact_signflip_p(d):
    """Two-sided exact paired permutation over all 2^n sign assignments.

    n=5 gives 32 arrangements, so the smallest attainable two-sided p is
    2/32 = 0.0625. That floor is a property of the design, not of the data.
    """
    obs = abs(d.mean())
    ge = sum(1 for sg in itertools.product([1, -1], repeat=len(d))
             if abs((d * np.array(sg)).mean()) >= obs - 1e-15)
    return ge / (2 ** len(d)), 2 ** len(d)


def exact_label_p(a, b):
    """Two-sided exact unpaired permutation over all C(n_a+n_b, n_a) splits."""
    pool = np.concatenate([a, b])
    n = len(pool)
    obs = abs(a.mean() - b.mean())
    idx = range(n)
    total = ge = 0
    for combo in itertools.combinations(idx, len(a)):
        total += 1
        mask = np.zeros(n, dtype=bool)
        mask[list(combo)] = True
        if abs(pool[mask].mean() - pool[~mask].mean()) >= obs - 1e-15:
            ge += 1
    return ge / total, total


def ols_slope(x, y):
    """Slope of y on x with its standard error. Quantifies the separability confound."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = ((x - xm) ** 2).sum()
    if sxx == 0 or n < 3:
        return None, None
    slope = ((x - xm) * (y - ym)).sum() / sxx
    resid = y - (ym + slope * (x - xm))
    se = float(np.sqrt((resid ** 2).sum() / (n - 2) / sxx))
    return float(slope), se


def contrast_block(cov, mean):
    """Paired and unpaired views of the same regime difference."""
    d = cov - mean
    m, lo, hi = t_ci(d)
    diff, wlo, whi, wdf = welch(cov, mean)
    sp, sn = exact_signflip_p(d)
    lp, ln = exact_label_p(cov, mean)
    return {
        "paired_t_ci": [float(m), float(lo), float(hi)],
        "paired_excludes_zero": bool(lo > 0 or hi < 0),
        "paired_dz": float(cohen_dz(cov, mean)),
        "unpaired_welch_ci": [diff, wlo, whi],
        "unpaired_welch_df": wdf,
        "unpaired_excludes_zero": bool(wlo > 0 or whi < 0),
        "exact_signflip_p": float(sp), "signflip_arrangements": sn,
        "exact_label_perm_p": float(lp), "label_arrangements": ln,
        "per_seed_difference": d.tolist(),
    }


# ---------- assembly --------------------------------------------------------

def analyse_run(run_dir):
    out = {}

    per_strength = {}
    for st in STRENGTHS:
        c, m = deltas(run_dir, "cov", st), deltas(run_dir, "mean", st)
        d = c - m
        mm, lo, hi = t_ci(d)
        per_strength[str(st)] = {
            "regime_difference_t_ci": [float(mm), float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "cov_delta_mean": float(c.mean()), "mean_delta_mean": float(m.mean()),
        }
    covers = [k for k, v in per_strength.items()
              if k != "0.0" and not v["excludes_zero"]]
    per_strength["_read_this_first"] = (
        "The pooled contrast below is NOT uniform across strengths. At "
        + ", ".join(f"s={k}" for k in covers)
        + " the regime difference COVERS ZERO"
          " -- and those are the strengths the mechanism predicts should be"
          " strongest. Do not quote the pooled figure without this."
    ) if covers else "Every non-zero strength resolves above zero."
    out["per_strength"] = per_strength

    cov_p = pooled_by_seed(run_dir, "cov", NONZERO)
    mean_p = pooled_by_seed(run_dir, "mean", NONZERO)
    out["pooled_contrast"] = contrast_block(cov_p, mean_p)
    out["pooled_contrast"]["cov_pooled_t_ci"] = [float(x) for x in t_ci(cov_p)]
    out["pooled_contrast"]["mean_pooled_t_ci"] = [float(x) for x in t_ci(mean_p)]

    # leave-one-seed-out on the pooled paired contrast
    jack = {}
    for drop in SEEDS:
        keep = [s for s in SEEDS if s != drop]
        c = pooled_by_seed(run_dir, "cov", NONZERO, keep)
        m = pooled_by_seed(run_dir, "mean", NONZERO, keep)
        mm, lo, hi = t_ci(c - m)
        jack[f"drop_{drop}"] = {"t_ci": [float(mm), float(lo), float(hi)],
                                "excludes_zero": bool(lo > 0 or hi < 0)}
    out["leave_one_seed_out"] = jack

    # specification robustness -- this contrast is a post-hoc pooling choice and
    # the paper polices forking paths, so its stability is reported, not assumed
    specs = {"nonzero_all4": NONZERO, "drop_s0.5": [0.25, 0.75, 1.0],
             "drop_s0.25": [0.5, 0.75, 1.0], "include_s0": STRENGTHS,
             "top2_only": [0.75, 1.0]}
    rob = {}
    for label, sts in specs.items():
        c = pooled_by_seed(run_dir, "cov", sts)
        m = pooled_by_seed(run_dir, "mean", sts)
        mm, lo, hi = t_ci(c - m)
        rob[label] = {"t_ci": [float(mm), float(lo), float(hi)],
                      "excludes_zero": bool(lo > 0 or hi < 0), "strengths": sts}
    out["specification_robustness"] = rob

    # separability confound: does a more separable cell simply show a bigger delta?
    slopes = {}
    for regime in ("cov", "mean"):
        xs, ys = [], []
        for st in STRENGTHS:
            for s in SEEDS:
                xs.append(load_f1(run_dir, regime, st, "gap", s))
                ys.append(load_f1(run_dir, regime, st, "gcp", s)
                          - load_f1(run_dir, regime, st, "gap", s))
        sl, se = ols_slope(xs, ys)
        slopes[regime] = {"slope_delta_on_gap_f1": sl, "se": se, "n": len(xs)}
    out["separability_slopes"] = slopes
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/gateB_synthetic_n40k/regime_contrast.json")
    args = ap.parse_args()

    payload = {}
    for label, d in RUNS.items():
        if not os.path.isdir(d):
            print(f"skip {label}: {d} missing")
            continue
        payload[label] = analyse_run(d)

    if "N40000" in payload and "N4000" in payload:
        a = pooled_by_seed(RUNS["N40000"], "cov", NONZERO) - \
            pooled_by_seed(RUNS["N40000"], "mean", NONZERO)
        b = pooled_by_seed(RUNS["N4000"], "cov", NONZERO) - \
            pooled_by_seed(RUNS["N4000"], "mean", NONZERO)
        diff, lo, hi, df = welch(a, b)
        payload["N40000_vs_N4000"] = {
            "welch_ci": [diff, lo, hi], "df": df,
            "excludes_zero": bool(lo > 0 or hi < 0),
            "_reading": ("If this covers zero the two sample sizes are NOT "
                         "distinguishable, and the N=4,000 arm is absence of "
                         "evidence rather than evidence of absence."),
        }

    payload["_headline"] = {
        "quote": "N40000.pooled_contrast.unpaired_welch_ci + exact_label_perm_p",
        "why": ("Pairing costs df 7.4->4 for a ~3 percent SE gain (across-regime "
                "seed correlation ~+0.065), so the paired interval is wider, and "
                "the paired sign-flip is pinned at the n=5 floor of 2/32=0.0625."),
        "mandatory_companion": "per_strength (two endpoints cover zero)",
        "not_recommended": {
            "paired_signflip_p": "cannot reach 0.05 at n=5 by construction",
            "rank_correlation_per_regime": ("the two rho intervals overlap; their "
                                            "bootstrapped difference crosses zero"),
        },
        "caveat_label_permutation": ("tests a stronger null than the sign-flip: it "
                                     "assumes a seed's cov and mean values are "
                                     "exchangeable across the group boundary"),
    }
    payload["_method"] = {
        "interval": "Student-t df=n-1; Welch for the unpaired contrast",
        "permutations": "fully enumerated, exact: 2^5=32 paired, C(10,5)=252 unpaired",
        "delta": "GCP minus GAP test macro-F1 at matched (regime, strength, seed)",
        "pooling": "per seed, mean over the four non-zero strengths",
        "source_cells": RUNS,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    for label in ("N40000", "N4000"):
        if label not in payload:
            continue
        p = payload[label]
        print(f"\n=== {label} ===")
        print("  per-strength regime contrast (cov minus mean), Student-t df=4:")
        for st in STRENGTHS:
            v = p["per_strength"][str(st)]
            m, lo, hi = v["regime_difference_t_ci"]
            print(f"    s={st:<5} {m:+.5f} [{lo:+.5f}, {hi:+.5f}]  "
                  f"{'excl 0' if v['excludes_zero'] else 'COVERS 0'}")
        print(f"    -> {p['per_strength']['_read_this_first']}")
        pc = p["pooled_contrast"]
        m, lo, hi = pc["paired_t_ci"]
        print(f"  pooled paired   {m:+.5f} [{lo:+.5f}, {hi:+.5f}] d_z={pc['paired_dz']:+.2f}")
        m, lo, hi = pc["unpaired_welch_ci"]
        print(f"  pooled unpaired {m:+.5f} [{lo:+.5f}, {hi:+.5f}] df={pc['unpaired_welch_df']:.2f}")
        print(f"  exact p: sign-flip {pc['exact_signflip_p']:.4f} "
              f"({pc['signflip_arrangements']} arr)   "
              f"label-perm {pc['exact_label_perm_p']:.4f} "
              f"({pc['label_arrangements']} arr)")
    if "N40000_vs_N4000" in payload:
        v = payload["N40000_vs_N4000"]
        d, lo, hi = v["welch_ci"]
        print(f"\n=== N=40k vs N=4k ===\n  {d:+.5f} [{lo:+.5f}, {hi:+.5f}]  "
              f"{'excl 0' if v['excludes_zero'] else 'COVERS 0'}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
