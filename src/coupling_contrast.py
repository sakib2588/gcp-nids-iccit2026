"""Inverse-Potts coupling contrast - Tier 2a.

WHAT THIS MODULE IS, AND WHAT IT IS NOT
--------------------------------------
This module fits a *regularised inverse correlation* (precision) matrix on a
gauge-fixed one-hot discretisation of NetFlow features, and contrasts the
precision matrix of an attack family against the benign one. That is the whole
computation. It is a discretised direct-coupling-style estimator of pairwise
couplings.

It is **not** an Energy-based Flow Classifier. It computes no energy, no local
fields, and it classifies nothing. Earlier revisions of this file cited
Pontes et al. 2021 (DOI 10.1109/TNSM.2021.3075503) as the method being
implemented; that citation was wrong and has been removed. The inverse-Potts
coupling estimate here is the same object DCA-style methods use as an
*ingredient*, but the surrounding EFC machinery (per-flow energy from couplings
and local fields, energy thresholding for classification) is absent.

WHAT IT MEASURES
----------------
An independent second-order view on NIDS flows, used to cross-validate the GCP
finding: "GCP helps a family only when that family's class separation lives in
feature covariance." For each attack family we ask how much of the family's
deviation-from-benign structure lives in *cross-feature* coupling (off-diagonal
blocks) versus *single-feature* marginal shift (diagonal blocks). The ratio

    coupling_contrast_index = off_mean / (off_mean + diag_mean)

is high for families whose signature is a change in feature co-occurrence
(relational: scanning, backdoor, injection) and low for families whose signature
is mostly a magnitude/marginal shift (volumetric: DoS/DDoS).

BLOCK-COUNT NORMALISATION (fixed 2026-07-20)
--------------------------------------------
``off`` and ``diag`` are means *per block*, not sums over blocks. There are
F(F-1) off-diagonal feature blocks and only F diagonal ones, so a plain sum
inflates the off-diagonal side by a factor of F-1 (38x at F=39). The pre-fix
code summed without dividing, which made every family read as "covariance
dominated" (index > 0.5) purely by counting. Because the correction is a
strictly monotone transform of the diag/off ratio, family *ranking* is
unaffected (Spearman 1.0 old vs new), so any conclusion resting on the ranking
or on the downstream correlation survives; only the absolute reading changes.

DEGENERATE-COLUMN HANDLING (fixed 2026-07-20)
---------------------------------------------
Quantile edges on heavily tied NetFlow features collapse: asking for 30 bins on
the real NF-ToN-IoT-v2 train split yields about 3 *distinct* edges for most
features, so ~90 percent of the nominal one-hot columns were identically zero.
``np.corrcoef`` returns NaN for a constant column, and the old code ran
``np.nan_to_num`` before inverting, which set a constant column's *diagonal*
entry to 0 rather than 1. That entry then equalled the ridge alone, and the
inverse produced exactly -1/ridge = -1000 on those columns, dominating the
coupling matrix with pure numerical artefact.

The fix is twofold:
  1. the per-feature bin count is taken from the number of *distinct* quantile
     edges, so ``n_bins`` is an upper request rather than a promise; and
  2. columns that are constant in either the benign reference rows or the rows
     being scored are dropped before the inversion, and the benign matrix is
     restricted to the same surviving basis so the two are differenced on a
     common support. No ``nan_to_num`` is applied; the reduced correlation
     matrix is NaN-free by construction.

Features left with no surviving column are excluded from both the block loop
and the block counts.

DIFFERENCING AGAINST BENIGN
---------------------------
The off/diag split is measured on the *deviation* from benign
(``e_family - e_benign``) rather than on a single fixed benign model: a benign
model has (by construction) no cross-feature couplings for features that are
independent under benign traffic, so its off-diagonal energy is zero regardless
of what is scored - a benign-only model literally cannot localise a family's
*new* correlation. Differencing the family's couplings against benign's cancels
the shared within-feature one-hot nuisance on the diagonal while exposing
exactly the covariance structure a family adds (off-diagonal) or the marginal
shift it adds (diagonal).

GAUGE FIXING
------------
Each flow is encoded as a gauge-fixed one-hot indicator: one column per
(feature, bin) with the last bin of every feature dropped. The full per-feature
one-hot columns sum to 1 and are therefore collinear, which makes the indicator
correlation matrix near-singular and lets the precision matrix's diagonal blocks
explode by ~2 orders of magnitude, swamping the real signal. Dropping a
reference state per feature is the standard Potts/DCA gauge that removes this
redundancy.

OOM SAFETY (past incident on this workstation)
----------------------------------------------
The one-hot indicator matrix is rows x sum_j (bins_j - 1). Building it over all
~2M flows would allocate tens of GB and freeze the machine. ``fit`` caps the
benign fit at ``max_fit_rows`` (default 80_000) and the scoring methods cap each
family at ``max_score_rows`` (default 40_000), both via a reproducible
``np.random.default_rng(42)`` subsample. No one-hot array over ~100k rows is
ever materialised.
"""
from __future__ import annotations

import numpy as np

_SUBSAMPLE_SEED = 42


class CouplingContrast:
    """Gauge-fixed inverse-Potts coupling contrast against a benign reference.

    Parameters
    ----------
    n_bins
        *Requested* quantile bins per feature. The realised count per feature is
        ``len(np.unique(edges)) - 1`` and is usually much smaller on tied
        NetFlow features; see ``effective_bins_``.
    ridge
        Ridge added to the correlation diagonal before inversion.
    max_fit_rows, max_score_rows
        Row caps for the benign fit and for each scored family.
    """

    def __init__(self, n_bins: int = 30, ridge: float = 1e-3,
                 max_fit_rows: int = 80_000, max_score_rows: int = 40_000):
        self.n_bins = n_bins
        self.ridge = ridge
        self.max_fit_rows = max_fit_rows
        self.max_score_rows = max_score_rows

    # ------------------------------------------------------------------ fit ---
    def fit(self, benign: np.ndarray) -> "CouplingContrast":
        benign = np.asarray(benign, dtype=np.float64)
        benign = self._subsample(benign, self.max_fit_rows)
        self._F = benign.shape[1]

        # Quantile bin edges per feature, fit on benign (shared grid). Ties in
        # the feature collapse edges, so the realised bin count is taken from
        # the number of DISTINCT edges rather than from n_bins.
        self._cuts = [self._bin_cuts(benign[:, j]) for j in range(self._F)]
        self._bins = [len(c) + 1 for c in self._cuts]
        # gauge-fixed columns per feature: one per bin minus the reference bin,
        # i.e. exactly the number of cuts. A constant feature has no cut and so
        # contributes no column.
        self._b = [n - 1 for n in self._bins]
        self._slices = []
        start = 0
        for width in self._b:
            self._slices.append(slice(start, start + width))
            start += width
        self._n_cols = start

        # Benign reference: keep the FULL correlation matrix (NaN columns and
        # all) so any later active-column subset can be sliced out of it, plus
        # the constant-column mask.
        oh = self._onehot(benign)
        self._C_benign, self._const_benign = self._corr(oh)
        return self

    # --------------------------------------------------------- introspection ---
    @property
    def effective_bins_(self) -> list:
        """Realised bin count per feature (from distinct quantile edges)."""
        return list(self._bins)

    @property
    def n_columns_(self) -> int:
        """Total gauge-fixed one-hot columns on the realised bin grid."""
        return self._n_cols

    # ------------------------------------------------------------- internals ---
    def _bin_cuts(self, col: np.ndarray) -> np.ndarray:
        """Interior quantile cut points, deduplicated.

        ``n_bins`` is a request. Heavily tied NetFlow features collapse most of
        the requested quantile edges onto the same value, and the realised bin
        count is ``len(cuts) + 1``. A feature with a single distinct value has
        no cut and is dropped downstream.
        """
        edges = np.unique(np.quantile(col, np.linspace(0, 1, self.n_bins + 1)))
        cuts = edges[1:-1]
        if cuts.size == 0 and edges.size >= 2:
            # two distinct edges only (e.g. a binary feature): the interior is
            # empty, so cut at the upper edge to keep the two states separable
            cuts = edges[1:]
        return cuts

    def _subsample(self, X: np.ndarray, cap: int) -> np.ndarray:
        if cap is not None and X.shape[0] > cap:
            rng = np.random.default_rng(_SUBSAMPLE_SEED)
            idx = rng.choice(X.shape[0], size=cap, replace=False)
            return X[idx]
        return X

    def _onehot(self, X: np.ndarray) -> np.ndarray:
        """Gauge-fixed one-hot indicator on the realised per-feature bin grid."""
        n = X.shape[0]
        oh = np.zeros((n, self._n_cols), dtype=np.float64)
        rows = np.arange(n)
        for j in range(self._F):
            width = self._b[j]
            if width == 0:                     # constant feature, no column
                continue
            idx = np.clip(np.digitize(X[:, j], self._cuts[j]), 0, self._bins[j] - 1)
            keep = idx < width                 # drop reference bin (gauge fixing)
            oh[rows[keep], self._slices[j].start + idx[keep]] = 1.0
        return oh

    @staticmethod
    def _corr(oh: np.ndarray):
        """Correlation matrix of the indicator columns plus a constant-column mask.

        Constant columns produce NaN rows/columns; they are returned as-is and
        are removed later by the active-column intersection, never patched with
        ``nan_to_num``.
        """
        const = oh.std(axis=0) == 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.corrcoef(oh, rowvar=False)
        return np.atleast_2d(C), const

    def _couplings_on(self, C: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Inverse-Potts couplings e = -(C + ridge*I)^{-1} on active columns."""
        sub = C[np.ix_(active, active)]
        if not np.all(np.isfinite(sub)):
            raise FloatingPointError(
                "non-finite entry in the reduced correlation matrix; the active"
                " column mask failed to remove a degenerate column")
        sub = sub + self.ridge * np.eye(sub.shape[0])
        return -np.linalg.inv(sub)

    def _active_mask(self, const_X: np.ndarray) -> np.ndarray:
        """Columns usable in BOTH the benign reference and the scored rows."""
        return ~(self._const_benign | const_X)

    # --------------------------------------------------------------- scoring ---
    def coupling_matrix(self, X: np.ndarray):
        """Couplings for rows X, restricted to columns shared with benign.

        Returns ``(e, active)`` where ``active`` is the boolean column mask the
        matrix is defined on.
        """
        X = np.asarray(X, dtype=np.float64)
        X = self._subsample(X, self.max_score_rows)
        C_x, const_x = self._corr(self._onehot(X))
        active = self._active_mask(const_x)
        return self._couplings_on(C_x, active), active

    def block_coupling_split(self, X: np.ndarray):
        """(off, diag) mean per-block coupling contrast of X against benign.

        Returns the *mean* absolute coupling contrast over cross-feature blocks
        (off-diagonal) and over single-feature blocks (diagonal) of
        ``e_X - e_benign``. Each side is divided by its own block count, so the
        two are directly comparable: ``off > diag`` means the rows differ from
        benign mainly in *feature covariance*, ``diag > off`` mainly in
        single-feature marginals.
        """
        d = self.block_coupling_detail(X)
        return d["off"], d["diag"]

    def coupling_contrast_index(self, X: np.ndarray) -> float:
        """off / (off + diag) on block-count-normalised means; NaN if degenerate."""
        return self.block_coupling_detail(X)["index"]

    def block_coupling_detail(self, X: np.ndarray) -> dict:
        """Full block breakdown of ``e_X - e_benign``.

        Keys: ``off``/``diag`` (block-count-normalised means, the values the
        index is built from), ``off_sum``/``diag_sum`` (raw sums over blocks,
        retained only so the pre-fix un-normalised quantity stays auditable),
        ``n_off_blocks``/``n_diag_blocks``, ``n_active_features``,
        ``n_active_columns`` and ``index``.
        """
        X = np.asarray(X, dtype=np.float64)
        X = self._subsample(X, self.max_score_rows)
        C_x, const_x = self._corr(self._onehot(X))
        active = self._active_mask(const_x)

        e_x = self._couplings_on(C_x, active)
        e_b = self._couplings_on(self._C_benign, active)
        delta = e_x - e_b

        # map each surviving column to its row/col position in the reduced matrix
        pos = np.cumsum(active) - 1
        feat_cols = []
        for j in range(self._F):
            sl = self._slices[j]
            cols = np.flatnonzero(active[sl.start:sl.stop])
            feat_cols.append(pos[sl.start + cols] if cols.size else np.empty(0, int))
        live = [j for j in range(self._F) if feat_cols[j].size > 0]

        off_sum = diag_sum = 0.0
        n_off = n_diag = 0
        for i in live:
            for j in live:
                blk = delta[np.ix_(feat_cols[i], feat_cols[j])]
                e = float(np.mean(np.abs(blk)))
                if i == j:
                    diag_sum += e
                    n_diag += 1
                else:
                    off_sum += e
                    n_off += 1

        off = off_sum / n_off if n_off else float("nan")
        diag = diag_sum / n_diag if n_diag else float("nan")
        denom = off + diag
        index = float(off / denom) if denom > 0 else float("nan")
        return {
            "off": off,
            "diag": diag,
            "off_sum": off_sum,
            "diag_sum": diag_sum,
            "n_off_blocks": n_off,
            "n_diag_blocks": n_diag,
            "n_active_features": len(live),
            "n_active_columns": int(active.sum()),
            "index": index,
        }
