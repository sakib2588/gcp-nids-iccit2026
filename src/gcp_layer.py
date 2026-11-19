"""T2 — Global Covariance Pooling (iSQRT-COV) adapted to 1D feature maps.

Reference: Li et al., "Towards Faster Training of Global Covariance Pooling
Networks by Iterative Matrix Square Root Normalization" (iSQRT-COV, CVPR 2018);
original Conv2d implementation jiangtaoxie/fast-MPN-COV.

Pipeline on a 1D conv feature map x of shape (B, C, L'):
  1. Covariance pooling  -> SPD matrix (B, C, C) over the L' positions
  2. Matrix square root via Newton-Schulz (3 iters, GPU-friendly, no SVD)
  3. Upper-triangular vectorisation -> (B, C*(C+1)/2)

Rank condition (plan v3 C.1 / Gap 6): `covariance_pool` CENTRES across the L'
positions, so it removes one degree of freedom and the covariance rank is
bounded by min(C, L'-1), not min(C, L'). The condition for a full-rank
covariance is therefore L' >= C + 1, and the deficiency test is L' < C + 1.
The equality case L' == C is deficient: at F=38 the CNN gives L'=32 and with
C=32 the covariance has rank 31 and a minimum eigenvalue near 5e-18.
(Before 2026-07-20 this module tested L' < C and so reported L' == C as
full-rank; see notes/20260720-bug-prepublication-code-audit.md, defect B7.)

When the condition is violated the square root can diverge to NaN. This module
GUARDS that case with a small ridge (eps*I) so the result stays finite, records
that the condition was violated (`last_rank_deficient`), and RAISES on any
non-finite output rather than silently sanitising it. The proper fix (reduce C
or raise L') is an architecture decision driven by Gap 6 - the guard only
prevents crashes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def covariance_pool(x: torch.Tensor) -> torch.Tensor:
    """(B, C, L') -> centered, L'-normalised covariance (B, C, C)."""
    B, C, L = x.shape
    # I_hat = (1/L) (I_L - (1/L) 1 1^T): centers across positions and divides by L
    eye = torch.eye(L, device=x.device, dtype=x.dtype)
    ones = torch.ones(L, L, device=x.device, dtype=x.dtype)
    i_hat = (1.0 / L) * eye - (1.0 / (L * L)) * ones        # (L, L)
    cov = x @ i_hat @ x.transpose(1, 2)                     # (B, C, C)
    return cov


def sqrtm_newton_schulz(A: torch.Tensor, iters: int = 3,
                        eps: float = 1e-5) -> torch.Tensor:
    """Batched SPD matrix square root via Newton-Schulz iteration."""
    B, C, _ = A.shape
    I = torch.eye(C, device=A.device, dtype=A.dtype).expand(B, C, C)
    # trace normalisation keeps the iteration in its convergence region
    trace = A.diagonal(dim1=1, dim2=2).sum(dim=1).view(B, 1, 1)
    trace = trace.clamp(min=eps)
    Y = A / trace
    Z = I.clone()
    for _ in range(iters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    return Y * torch.sqrt(trace)


class GlobalCovariancePooling(nn.Module):
    """iSQRT-COV pooling head. Output dim = C*(C+1)/2.

    Args:
        channels: C, the conv feature-map channel count (sets output dim).
        iters:    Newton-Schulz iterations. iSQRT-COV's own headline config uses
                  5; we use 3 (within 0.3% error per their ablation) given our
                  much smaller C=32-channel covariance and run-count budget.
        ridge:    eps added to the covariance diagonal for numerical stability;
                  also lifts rank when L' < C + 1.
        raise_on_nonfinite:
                  when True (default) a non-finite pooled output aborts the run
                  with RuntimeError. The old behaviour was to nan_to_num the
                  forward output only, which left the INPUT GRADIENT non-finite;
                  clip_grad_norm_ in src/engine.py then multiplied every
                  parameter gradient by a NaN total norm and destroyed the model
                  while the loss stayed finite. Set False only to inspect that
                  failure mode in tests.
    """

    def __init__(self, channels: int, iters: int = 3, ridge: float = 1e-5,
                 raise_on_nonfinite: bool = True):
        super().__init__()
        self.channels = channels
        self.iters = iters
        self.ridge = ridge
        self.raise_on_nonfinite = raise_on_nonfinite
        self.out_dim = channels * (channels + 1) // 2
        idx = torch.triu_indices(channels, channels)
        self.register_buffer("triu_i", idx[0], persistent=False)
        self.register_buffer("triu_j", idx[1], persistent=False)
        self.last_rank_deficient = False
        # running count of non-finite outputs (Gap 6). Persistent buffer so the
        # count survives state_dict / load_state_dict round-trips.
        self.register_buffer("nan_events", torch.zeros((), dtype=torch.long),
                             persistent=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Tolerate checkpoints written before nan_events became a buffer.

        Every checkpoint trained before 2026-07-20 lacks the `nan_events` key.
        Without this shim a strict load of those artifacts would fail, so the
        published checkpoints would stop evaluating.
        """
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)
        key = prefix + "nan_events"
        if key in missing_keys:
            missing_keys.remove(key)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L')
        B, C, L = x.shape
        assert C == self.channels, f"expected C={self.channels}, got {C}"
        # centring costs one degree of freedom: rank <= min(C, L-1), so the
        # full-rank condition is L >= C + 1 and the deficiency test is L < C + 1
        self.last_rank_deficient = bool(L < C + 1)

        cov = covariance_pool(x)                                # (B, C, C)
        # ridge: stabilise + lift rank when L' < C + 1
        eye = torch.eye(C, device=x.device, dtype=x.dtype)
        cov = cov + self.ridge * eye

        sqrt_cov = sqrtm_newton_schulz(cov, self.iters, self.ridge)
        vec = sqrt_cov[:, self.triu_i, self.triu_j]            # (B, C*(C+1)/2)
        # count non-finite rows BEFORE the backstop (Gap 6 NaN-event rate)
        n_bad = int((~torch.isfinite(vec)).any(dim=1).sum().item())
        if n_bad:
            self.nan_events += n_bad
            msg = (f"GlobalCovariancePooling produced {n_bad} non-finite "
                   f"row(s) of {B} (C={C}, L'={L}, "
                   f"rank_deficient={self.last_rank_deficient}). "
                   f"nan_events={int(self.nan_events)}. Sanitising the forward "
                   f"output does NOT sanitise the backward pass: the input "
                   f"gradient stays non-finite and clip_grad_norm_ then scales "
                   f"every parameter gradient by a NaN norm, silently "
                   f"destroying the model while the loss looks finite. "
                   f"Aborting instead.")
            if self.raise_on_nonfinite:
                raise RuntimeError(msg)
            # legacy path, kept only so tests can exhibit the failure mode
            print(f"[gcp_layer][WARNING] {msg}", flush=True)
            # hard backstop: never let a NaN/Inf escape into the classifier
            vec = torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec
