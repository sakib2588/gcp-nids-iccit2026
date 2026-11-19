"""Gap 1 - decision-threshold calibration on a target slice (plan v3 D.1 R1-7).

Under cross-dataset label shift, the 0.5 threshold is arbitrary. Pick the
threshold on p(attack) that maximises macro-F1 on a small target slice. The
caller MUST keep that slice disjoint from the final evaluation set.

Search space
------------
The default candidate set is the sorted unique predicted probabilities plus one
sentinel just above the maximum. That set is exact: as a threshold t sweeps the
real line, the predicted-label vector (p >= t) only changes when t crosses an
observed probability, so every partition of the slice that any threshold can
produce is reachable from this finite set, and no partition is missed. A fixed
grid such as ``np.linspace(0.05, 0.95, 19)`` is not exact: it cannot express any
cut below 0.05 or above 0.95, so whenever the optimum lies outside that window
the search returns a grid endpoint and the reported F1 is a lower bound on the
true calibrated F1. A log-spaced grid down to 1e-4 would widen the window but
would still be an approximation with an arbitrary floor, so the exact
unique-probability set is preferred.

Cost
----
Scoring every unique probability with a naive loop over ``f1_score`` would be
O(n^2). ``_macro_f1_at`` instead sorts once and reads confusion counts off
suffix sums, giving O(n log n) for the whole sweep. It reproduces
``sklearn.metrics.f1_score(y, pred, average="macro", zero_division=0)`` exactly,
including scikit-learn's rule that the averaged label set is the union of the
labels present in y_true and y_pred.
"""
from __future__ import annotations

import numpy as np

# The truncated grid used by the original implementation. Kept so regression
# tests can compare against it; not used by default any more.
LEGACY_GRID = np.linspace(0.05, 0.95, 19)


def _macro_f1_at(y, p, thresholds, order=None):
    """Vectorised macro-F1 of ``(p >= t)`` for every t in ``thresholds``.

    Matches ``f1_score(y, (p >= t).astype(int), average="macro",
    zero_division=0)`` element for element.
    """
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p, dtype=np.float64)
    t = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
    n = y.shape[0]

    if order is None:
        order = np.argsort(p, kind="stable")
    ps = p[order]
    ys = y[order]

    # suffix_pos[i] = number of true positives among the i-th and later samples
    # in probability order, so a cut at index i has TP = suffix_pos[i].
    suffix_pos = np.concatenate(([0.0], np.cumsum(ys[::-1], dtype=np.float64)))[::-1]

    idx = np.searchsorted(ps, t, side="left")
    n_pred_pos = (n - idx).astype(np.float64)
    tp = suffix_pos[idx]
    fp = n_pred_pos - tp

    n_pos = float(ys.sum())
    n_neg = float(n - n_pos)
    fn = n_pos - tp
    tn = n_neg - fp

    den_pos = 2.0 * tp + fp + fn
    den_neg = 2.0 * tn + fn + fp
    f1_pos = np.divide(2.0 * tp, den_pos, out=np.zeros_like(den_pos),
                       where=den_pos > 0)
    f1_neg = np.divide(2.0 * tn, den_neg, out=np.zeros_like(den_neg),
                       where=den_neg > 0)

    if n_pos > 0 and n_neg > 0:
        # Both labels appear in y_true, so scikit-learn averages over {0, 1}
        # regardless of what the prediction contains.
        return 0.5 * (f1_pos + f1_neg)

    # Degenerate slice: y_true is single-class, so the averaged label set is
    # {c} when the prediction is also all-c and {0, 1} otherwise.
    both = 0.5 * (f1_pos + f1_neg)
    if n == 0:
        return both
    if n_pos == 0:
        single = (n_pred_pos == 0)          # prediction is all-0 like y_true
        return np.where(single, f1_neg, both)
    single = (n_pred_pos == n)              # prediction is all-1 like y_true
    return np.where(single, f1_pos, both)


def candidate_thresholds(p_attack):
    """Exact candidate set: sorted unique probabilities plus a top sentinel.

    The sentinel is the next representable float above the maximum, which is
    the only way to express the all-negative partition.
    """
    p = np.asarray(p_attack, dtype=np.float64)
    uniq = np.unique(p)
    if uniq.size == 0:
        return uniq
    return np.append(uniq, np.nextafter(uniq[-1], np.inf))


def calibrate_threshold(y_true, p_attack, grid=None):
    """Return (thr, f1_at_thr, f1_at_0.5), all on the supplied slice.

    ``grid=None`` runs the exact unique-probability search described in the
    module docstring. Passing an explicit ``grid`` restricts the search to that
    set, which is what the regression tests use to reproduce the old behaviour.
    Ties are broken toward the lowest threshold, as in the original loop.
    """
    y_true = np.asarray(y_true)
    p = np.asarray(p_attack, dtype=np.float64)

    order = np.argsort(p, kind="stable")
    f1_05 = float(_macro_f1_at(y_true, p, 0.5, order=order)[0])

    cand = candidate_thresholds(p) if grid is None else np.asarray(
        grid, dtype=np.float64)
    if cand.size == 0:
        return 0.5, f1_05, f1_05
    cand = np.sort(cand)

    scores = _macro_f1_at(y_true, p, cand, order=order)
    best = int(np.argmax(scores))          # argmax takes the first maximum
    return float(cand[best]), float(scores[best]), f1_05
