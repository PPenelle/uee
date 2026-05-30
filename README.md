# Uniform Error Estimator (UEE)

UEE is a Python implementation of minimax (Chebyshev) regression.

It estimates a model that minimizes the worst-case prediction error instead of the average error.

FEATURES

- Multivariate regression
- Active set detection
- Dual weights (influence)
- Robustness measures
- Bootstrap inference

INSTALLATION

Clone the repository:

git clone https://github.com/PPENELLE/uee.git

EXAMPLE

from uee import UEE
import numpy as np

X = np.array([100, 200, 300, 400])
y = np.array([120, 180, 260, 310])

model = UEE()
res = model.fit(X, y)

print(res.params)

INTERPRETATION

Active set = observations driving the model  
Dual weights = influence of each observation  

LICENSE

MIT License
