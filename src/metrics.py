"""T4 — statistics for the GCP-NIDS paper (plan v3 C.3).

Protocol: bootstrap 95% CI for per-cell precision; Cohen's d_z for paired
GCP-vs-GAP effect size; per-family F1; DESCRIPTIVE across-dataset ranking
(NO Friedman/Wilcoxon at n=4 datasets — invalid, see plan C.3).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score


def bootstrap_ci(values, n_boot: int = 10_000, ci: float = 0.95,
                 seed: int = 42):
    """Percentile bootstrap CI over a small sample (e.g. 5 per-seed scores).

    Returns (mean, lo, hi). Constant input -> zero-width CI.
    """
    v = np.asarray(values, dtype=np.float64)
    assert v.ndim == 1 and len(v) >= 1
    rng = np.random.default_rng(seed)
    if np.allclose(v, v[0]):
        return float(v[0]), float(v[0]), float(v[0])
    boot = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo = float(np.percentile(boot, 100 * (1 - ci) / 2))
    hi = float(np.percentile(boot, 100 * (1 + ci) / 2))
    return float(v.mean()), lo, hi


def t_ci(values, ci: float = 0.95):
    """Student-t CI for the mean of a small sample. Returns (mean, lo, hi).

    Use this, not `bootstrap_ci`, whenever the inferential claim is "the
    interval excludes zero" at n around 5. A percentile bootstrap of a mean
    resamples the observed points, so its endpoints can never fall outside
    [min(values), max(values)]: at n=5 the interval is structurally too short.
    Simulated coverage over 3,000 draws of n=5 standard normals is 0.839 for
    `bootstrap_ci` against 0.946 for this function, so a bootstrap "excludes
    zero" verdict carries a false-positive rate near 16 percent, not 5.

    `bootstrap_ci` is deliberately left in place and unmodified: the published
    results on disk were produced with it, and the two must stay comparable.

    Interval is mean +/- t_{(1+ci)/2, n-1} * sd / sqrt(n), sd with ddof=1.
    Constant input -> zero-width interval (matching `bootstrap_ci`).
    A single point has no df and returns an infinite interval.
    """
    v = np.asarray(values, dtype=np.float64)
    assert v.ndim == 1 and len(v) >= 1
    mean = float(v.mean())
    n = len(v)
    if n < 2:
        return mean, float("-inf"), float("inf")
    sd = float(v.std(ddof=1))
    if sd == 0.0:
        return mean, mean, mean
    half = float(stats.t.ppf((1.0 + ci) / 2.0, n - 1)) * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def cohen_dz(a, b):
    """Paired effect size d_z = mean(a-b) / std(a-b, ddof=1).

    Positive => a > b on average.

    Zero paired SD is NOT no-effect. If every seed shows the identical nonzero
    gain the effect is maximally consistent and d_z is unbounded, so this
    returns +inf (or -inf for an identical loss). The old guard returned 0.0
    there, which downstream gates such as `dz >= 0.8` read as no effect,
    recording a perfect result as unsupported. Only an all-zero difference
    vector, no effect and no variance, returns 0.0.

    Callers that serialise the result must handle non-finite values; use
    `math.isfinite(dz)` and record the sign explicitly rather than writing a
    bare `Infinity` token into JSON.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1 and len(a) >= 2
    diff = a - b
    sd = diff.std(ddof=1)
    mean = float(diff.mean())
    if sd == 0:
        return 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
    return float(mean / sd)


def per_family_f1(y_true, y_pred, labels=None):
    """Per-class F1 as {class_label: f1}. No averaging, no accuracy."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    scores = f1_score(y_true, y_pred, labels=labels, average=None,
                      zero_division=0)
    return {lab: float(s) for lab, s in zip(labels, scores)}


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def effective_rank(X: np.ndarray) -> float:
    """Effective rank of a feature matrix (Roy & Vetterli 2007).

    erank = exp(H(p)), p_i = lambda_i / sum(lambda), lambda = eigenvalues of
    the feature covariance matrix. High erank (near n_features) => features
    carry rich, mutually-informative structure. Low erank (near 1) => features
    are near-independent / redundant, i.e. little for covariance pooling to
    exploit beyond first-order (mean) statistics.
    """
    X = np.asarray(X, dtype=np.float64)
    assert X.ndim == 2 and X.shape[0] >= 2
    cov = np.cov(X, rowvar=False)
    eigvals = np.linalg.eigvalsh(np.atleast_2d(cov))
    eigvals = np.clip(eigvals, 0.0, None)
    total = eigvals.sum()
    if total <= 0:
        return 1.0
    p = eigvals / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def eigen_summary(cov: np.ndarray, tol_rel: float = 1e-8):
    """Effective rank + condition number of a pre-built (C, C) covariance.

    Companion to `effective_rank`, which takes a raw (n, features) matrix and
    forms its covariance internally. Here `cov` is already the SPD matrix the
    GCP head operates on (mean of `covariance_pool(x)` over samples), so we
    eigendecompose it directly.

    Returns (effective_rank, condition_number):
      - effective_rank = exp(H(p)), p_i = lambda_i / sum(lambda) over the
        non-negative eigenvalues (Roy & Vetterli 2007). Saturates near
        min(C, rank(cov)); for the 1D-CNN covariance rank <= min(C, L'-1).
      - condition_number = lambda_max / lambda_min, where lambda_min is the
        smallest eigenvalue above tol_rel * lambda_max. A structurally
        rank-deficient covariance (L' < C) therefore reports a large but
        FINITE condition number driven by the numerical noise floor, not inf.
    """
    cov = np.asarray(cov, dtype=np.float64)
    assert cov.ndim == 2 and cov.shape[0] == cov.shape[1]
    w = np.linalg.eigvalsh(cov)
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0:
        return 1.0, 1.0
    p = w / total
    p = p[p > 0]
    erank = float(np.exp(-np.sum(p * np.log(p))))
    wmax = float(w.max())
    floor = tol_rel * wmax
    pos = w[w > floor]
    cond = float(wmax / pos.min()) if pos.size else 1.0
    return erank, cond


def descriptive_rank(model_scores: dict):
    """Across-dataset descriptive ranking (replaces invalid Friedman at n=4).

    Args:
        model_scores: {model_name: [score_per_dataset, ...]} aligned by dataset.
    Returns:
        {model_name: {"mean_rank": r, "wins": k, "n_datasets": n}}
        rank 1 = best on a dataset (higher score = better).
    """
    names = list(model_scores)
    arr = np.asarray([model_scores[n] for n in names], dtype=np.float64)  # (M, D)
    M, D = arr.shape
    out = {n: {"mean_rank": 0.0, "wins": 0, "n_datasets": D} for n in names}
    ranks = np.zeros((M, D))
    for d in range(D):
        order = np.argsort(-arr[:, d])            # best first
        col_rank = np.empty(M)
        col_rank[order] = np.arange(1, M + 1)
        ranks[:, d] = col_rank
        out[names[int(order[0])]]["wins"] += 1
    for i, n in enumerate(names):
        out[n]["mean_rank"] = float(ranks[i].mean())
    return out
