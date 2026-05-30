# Uniform Error Estimator (UEE)

UEE is a Python implementation of minimax (Chebyshev) regression.

It estimates a regression model that minimizes the worst-case (maximum) prediction error instead of the average squared error used in ordinary least squares (OLS).

------------------------------------------------------------

OVERVIEW

The UEE estimator solves:

min beta max |y - X beta|

The solution minimizes the largest absolute residual across all observations.

Unlike OLS, which uses all data points, UEE is typically determined by a small number of extreme observations (the active set).

------------------------------------------------------------

FEATURES

- Works with univariate and multivariate regressors
- Optional intercept (included by default)
- Exact minimax regression via linear programming
- Active set detection (binding observations)
- Dual weights (influence diagnostics)
- Influence / sensitivity measures
- Alternation check (Chebyshev condition)
- Bootstrap inference with confidence intervals
- Full input validation and error handling

------------------------------------------------------------

INSTALLATION

Clone the repository:

git clone https://github.com/YOURUSERNAME/uee.git

Requires:
- numpy
- scipy

------------------------------------------------------------

BASIC USAGE

from uee import UEE
import numpy as np

X = np.array([100, 200, 300, 400])
y = np.array([120, 180, 260, 310])

model = UEE()
res = model.fit(X, y)

print("Parameters:", res.params)
print("Max error:", res.max_abs_error)

------------------------------------------------------------

MODEL SPECIFICATION

With intercept (default)

model = UEE()

Model form:
y = beta_0 + beta_1 x

Without intercept

model = UEE(fit_intercept=False)

Model form:
y = beta_1 x

Note:
Disabling the intercept forces the regression through the origin and changes the geometry of the problem.

------------------------------------------------------------

OUTPUT FIELDS

After fitting:

res = model.fit(X, y)

Core results:

res.params            -> estimated coefficients
res.resid             -> residuals
res.fittedvalues      -> predicted values
res.max_abs_error     -> maximum absolute residual
res.ssr               -> sum of squared residuals

Structural diagnostics:

res.active_set        -> indices of binding observations
res.n_active          -> number of active points
res.rank_active       -> rank of active matrix
res.alternation       -> Chebyshev alternation condition

Influence diagnostics:

res.dual_weights      -> raw dual weights
res.dual_weights_norm -> normalized weights (sum = 1)

Interpretation:
- non-zero weight = influential observation
- zero weight = no influence

Robustness / sensitivity:

res.influence_bounds  -> sensitivity measure

------------------------------------------------------------

PREDICTION

y_pred = model.predict(X_new)

------------------------------------------------------------

BOOTSTRAP INFERENCE

boot = model.bootstrap(X, y, n_boot=200, random_state=42)

Returns:

boot["mean"]      -> mean of parameters
boot["std"]       -> standard deviation
boot["ci_lower"]  -> lower 95% bound
boot["ci_upper"]  -> upper 95% bound

------------------------------------------------------------

FULL EXAMPLE

from uee import UEE
import numpy as np

S = np.array([100, 200, 300, 400, 500])
C = np.array([120, 180, 260, 310, 360])

model = UEE()
res = model.fit(S, C)

print("Parameters:", res.params)
print("Max error:", res.max_abs_error)
print("Active set:", res.active_set)
print("Dual weights:", res.dual_weights_norm)

pred = model.predict(S)
print("Predictions:", pred)

boot = model.bootstrap(S, C, n_boot=100, random_state=42)
print("Bootstrap mean:", boot["mean"])

------------------------------------------------------------

INTERPRETATION

Active set:
The estimator is determined by a small set of observations (typically k+1 points).

Dual weights:
Measure the influence of observations. Only active points typically have non-zero weights.

Geometry:
The estimator finds the line that minimizes the maximum vertical deviation across all data points.

------------------------------------------------------------

IMPORTANT NOTES

- The estimator is sparse: only a few observations determine the solution
- Results can be sensitive to these observations
- Bootstrap variability can be high due to this sparsity
- UEE prioritizes worst-case accuracy, not average fit

------------------------------------------------------------

COMPARISON WITH OLS

OLS:
- minimizes squared error
- uses all observations
- optimized for average performance

UEE:
- minimizes maximum error
- depends on extreme observations
- optimized for worst-case performance

------------------------------------------------------------

LICENSE

MIT License
