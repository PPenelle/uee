# -*- coding: utf-8 -*-
"""
===============================================================================
UEE: Uniform Error Estimator (Minimax Regression)
-------------------------------------------------------------------------------
Author: Philippe G. Penelle, Ph.D.
License: MIT
Version: 1.0.1
NOTE:
# Active set identification requires a scale-aware tolerance.
# Fixed in v1.0.1 to ensure robustness to solver precision.
===============================================================================
This module implements the Uniform Error Estimator (UEE), an extended version
of classical Chebyshev (minimax) regression:
    min_β max_t |y_t - X_t β|
The implementation includes structural diagnostics (active set, dual weights,
alternation), robustness metrics, and bootstrap inference.
-------------------------------------------------------------------------------
LICENSE (MIT)
-------------------------------------------------------------------------------
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
===============================================================================
"""
import numpy as np
from scipy.optimize import linprog
# =============================================================================
# RESULTS OBJECT
# =============================================================================
class UEEResults:
    """
    Stores outputs and diagnostics from the UEE model.

    Attributes
    ----------
    params : ndarray
        Estimated coefficients.
    resid : ndarray
        Residuals.
    fittedvalues : ndarray
        Predicted values.
    max_abs_error : float
        Maximum absolute residual.
    ssr : float
        Sum of squared residuals.
    active_set : ndarray
    Indices of binding observations (|u_t| = E*), identified using
    a scale-aware tolerance to ensure robustness to numerical precision.
    dual_weights : ndarray
        Raw dual weights.
    dual_weights_norm : ndarray
        Normalized dual weights (sum to 1).
    influence_bounds : ndarray
        Sensitivity / influence measure.
    """
    def __init__(self, beta, E, residuals, X, lp_result):

        self.params = beta
        self.resid = residuals
        self.fittedvalues = X @ beta
        self.max_abs_error = float(E)
        self.n, self.m = X.shape
        self.abs_resid = np.abs(residuals)
        self.ssr = float(np.sum(residuals**2))
        # Active set
        # Active set (scale-aware tolerance)
        tol = 1e-6 * max(1.0, abs(E))
        self.active_set = np.where(
            np.abs(self.abs_resid - E) <= tol
        )[0]
        self.n_active = len(self.active_set)
        # Alternation condition
        if self.n_active > 0:
            s = np.sign(residuals[self.active_set])
            self.alternation = (np.any(s > 0) and np.any(s < 0))
        else:
            self.alternation = False
        # Rank condition
        try:
            if self.n_active >= self.m:
                self.rank_active = np.linalg.matrix_rank(X[self.active_set])
            else:
                self.rank_active = 0
        except Exception:
            self.rank_active = None
        self._compute_dual(lp_result)
        self._compute_influence_bounds()

    def _compute_dual(self, res):
        try:
            duals = res.ineqlin.marginals
            weights = []

            for i in range(0, len(duals), 2):
                w = abs(duals[i]) + abs(duals[i + 1])
                weights.append(w)

            weights = np.array(weights)
            self.dual_weights = weights

            total = np.sum(weights)
            if total > 0:
                self.dual_weights_norm = weights / total
            else:
                self.dual_weights_norm = weights

        except Exception:
            self.dual_weights = None
            self.dual_weights_norm = None

    def _compute_influence_bounds(self):
        if self.dual_weights_norm is None:
            self.influence_bounds = None
            return

        self.influence_bounds = self.dual_weights_norm.copy()
# =============================================================================
# MAIN ESTIMATOR
# =============================================================================
class UEE:
    """
    Uniform Error Estimator (UEE).

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept.

    Notes
    -----
    Disabling the intercept forces the regression through the origin and
    fundamentally changes the geometry of the minimax problem.
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self._fitted = False

    def fit(self, X, y):
        """
        Fit the UEE minimax regression model.

        Parameters
        ----------
        X : array-like
        y : array-like

        Returns
        -------
        UEEResults
        """

        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            raise ValueError("y must be 1D")

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, m = X.shape

        if len(y) != n:
            raise ValueError("Mismatch between X and y")

        if n < m + 1:
            raise ValueError("Underidentified model")

        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("NaNs detected")

        if self.fit_intercept:
            X = np.column_stack([np.ones(n), X])
            m += 1

        # LP setup
        n_vars = m + 1
        c = np.zeros(n_vars)
        c[-1] = 1.0

        A = []
        b = []

        for t in range(n):
            xt = X[t]

            row1 = np.zeros(n_vars)
            row1[:-1] = xt
            row1[-1] = -1

            row2 = np.zeros(n_vars)
            row2[:-1] = -xt
            row2[-1] = -1

            A.append(row1)
            b.append(y[t])

            A.append(row2)
            b.append(-y[t])

        bounds = [(None, None)] * m + [(0, None)]

        res = linprog(
            c,
            A_ub=np.array(A),
            b_ub=np.array(b),
            bounds=bounds,
            method="highs"
        )

        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")

        beta = res.x[:-1]
        E = res.x[-1]
        residuals = y - X @ beta

        self._results = UEEResults(beta, E, residuals, X, res)
        self._fitted = True

        return self._results

    def predict(self, X_new):
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        X_new = np.asarray(X_new, dtype=float)

        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)

        if self.fit_intercept:
            X_new = np.column_stack([np.ones(len(X_new)), X_new])

        return X_new @ self._results.params

    def bootstrap(self, X, y, n_boot=200, random_state=None):
        """
        Bootstrap inference for UEE parameters.
        """

        rng = np.random.default_rng(random_state)

        X = np.asarray(X)
        y = np.asarray(y)

        n = len(y)
        samples = []

        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)

            try:
                res = UEE(self.fit_intercept).fit(X[idx], y[idx])
                samples.append(res.params)
            except Exception:
                continue

        if len(samples) == 0:
            raise RuntimeError("Bootstrap failed")

        samples = np.array(samples)

        return {
            "mean": samples.mean(axis=0),
            "std": samples.std(axis=0),
            "ci_lower": np.percentile(samples, 2.5, axis=0),
            "ci_upper": np.percentile(samples, 97.5, axis=0)
        }
