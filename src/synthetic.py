"""T5 -- synthetic covariance-signal vs mean-signal data generator (GATE B).

Controlled test of the theory in paper_writing/theory_gcp_failure.md: GCP
should only beat GAP when class separation actually lives in feature-pair
covariance structure, not in per-feature means. Two knobbed regimes:

- "mean": classes share one covariance matrix; separation is a mean shift.
  Covariance pooling has nothing extra to find here -- GAP should win or tie.
- "cov": classes share the same per-feature mean and variance (marginals
  match), but differ in the OFF-diagonal correlation structure. GAP is blind
  to this; GCP should win as the knob strengthens.

`strength` in [0, 1] controls signal magnitude (0 = classes identical /
chance-level; 1 = maximally separated regime).
"""
from __future__ import annotations

import numpy as np


# Receptive field of the 3-layer, kernel-3, unpadded backbone is 7, so pairs at
# lag <= 6 share a window. FAR_NORM is the spectral norm of the off-diagonal
# part (< 1 keeps the correlation matrix positive definite).
FAR_BAND = 6
FAR_NORM = 0.97


def _shared_base_covariance(n_features: int, rng: np.random.Generator) -> np.ndarray:
    """A mild, realistic base covariance (weakly correlated, PSD, unit diag)."""
    A = rng.normal(scale=0.15, size=(n_features, n_features))
    cov = A @ A.T
    d = np.sqrt(np.diag(cov))
    cov = cov / np.outer(d, d)
    np.fill_diagonal(cov, 1.0)
    return cov


def generate_synthetic(
    n_samples: int = 4000,
    n_features: int = 39,
    signal: str = "cov",
    strength: float = 1.0,
    seed: int = 42,
):
    """Generate a synthetic binary classification dataset.

    Args:
        n_samples: total samples, split 50/50 across the two classes.
        n_features: feature dimensionality (39 matches NF-UQ-NIDS-v2).
        signal: "mean" -> classes differ only in per-feature mean.
                "cov"  -> classes differ only in feature-pair covariance
                          (means and variances matched across classes).
        strength: knob in [0, 1]; 0 = no class separation, 1 = strong.
        seed: RNG seed.

    Returns:
        X: (n_samples, n_features) float64 array.
        y: (n_samples,) int array in {0, 1}.
    """
    assert signal in ("mean", "cov", "cov_far")
    assert 0.0 <= strength <= 1.0
    rng = np.random.default_rng(seed)
    n0 = n_samples // 2
    n1 = n_samples - n0

    base_cov = _shared_base_covariance(n_features, rng)

    if signal == "mean":
        cov0 = cov1 = base_cov
        mu0 = np.zeros(n_features)
        direction = rng.normal(size=n_features)
        direction /= np.linalg.norm(direction)
        mu1 = direction * (strength * 2.5)
    elif signal == "cov_far":
        # Reviewer run 2026-11-18 (R1.3). The conv backbone reads local
        # windows (receptive field 7), so any class difference in correlation
        # between NEARBY features leaks into per-window variance and then,
        # through the ReLU, into first-order pooled statistics. Here both
        # classes have identity correlation inside the band |i-j|<=BAND, so
        # every receptive-field window has an identical distribution in both
        # classes. Classes differ only in correlation between FAR feature
        # pairs, which only a pooled second-order statistic across positions
        # can see. Unit variance and zero mean in both classes.
        mu0 = mu1 = np.zeros(n_features)
        ii, jj = np.indices((n_features, n_features))
        far = (np.abs(ii - jj) > FAR_BAND)

        def _far_corr(r):
            w = np.triu(r.uniform(-1.0, 1.0, size=(n_features, n_features)), 1)
            w = (w + w.T) * far
            w *= FAR_NORM / np.linalg.norm(w, 2)
            return np.eye(n_features) + w

        w0, w1 = _far_corr(rng), _far_corr(rng)
        cov0 = w0
        cov1 = (1 - strength) * w0 + strength * w1
    else:  # "cov"
        mu0 = mu1 = np.zeros(n_features)
        cov0 = base_cov
        # Boost off-diagonal correlation strength for class 1 only, holding
        # unit variance (diagonal) fixed so per-feature marginals match.
        extra = _shared_base_covariance(n_features, rng)
        cov1 = (1 - strength) * base_cov + strength * extra
        np.fill_diagonal(cov1, 1.0)
        cov1 = _nearest_psd(cov1)

    X0 = rng.multivariate_normal(mu0, cov0, size=n0)
    X1 = rng.multivariate_normal(mu1, cov1, size=n1)
    X = np.vstack([X0, X1]).astype(np.float64)
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def _nearest_psd(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Clip negative eigenvalues so a correlation-like matrix stays valid PSD."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, eps, None)
    fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed
