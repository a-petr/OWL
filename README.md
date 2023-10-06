# (OWL) Orthogonally Weighted L_{2,1}  regularizer

Multi-task Lasso model trained with Orthogonally Weighted L_{2,1} (OWL) regularizer.
The model optimizes the following objective function:

$$ 
\frac{1}{2 \alpha} ||Y - AZ||\_{\text{Fro}}^2 + ||Z(Z^TZ)^{-1/2}||\_{2,1}
$$

where $||Z||\_{2,1}$ is defined as the sum of norms of each row: $\sum_i \sqrt{\sum\_j Z\_{ij}^2}$. 

The class `OrthogonallyWeightedL21` implements a direct variable metric proximal gradient iterative minimization algorithm with optional choice of $\alpha$ based on a discrepancy principle.
The class `OrthogonallyWeightedL21Continuation` implements a numerical continuation strategy for the relaxed problem

$$ 
\frac{1}{2 \alpha} ||Y - AZ||\_{\text{Fro}}^2 + ||Z(\gamma I + (1-\gamma)Z^TZ)^{-1/2}||\_{2,1}
$$

for a sequence of $\gamma \in (0,1]$ with $\gamma \to 0$ to improve robustness.

For more information on this model, please refer to "Orthogonally weighted $L_{2,1}$ regularization for rank-aware joint sparse recovery: algorithm and analysis" by A. Petrosyan, K. Pieper, H. Tran.



## Parameters

- `alpha` (float, default=None):
  Regularization parameter $\alpha$.
  Defaults to None in which case a heuristic formula is used.
  If noise_level is set, the value will only be used for initialization, and then adapted according to a discrepancy principle.
- `noise_level` (float, default=None):
  Estimated noise level $\delta$.
  If set to a positive value, $\alpha$ will be adapted until $\tau_1 \delta \leq ||Y - AX|| \leq \tau_2 \delta$.
- `max_iter` (int, default=1000):
  The maximum number of iterations, including failed updates.
- `tol` (float, default=1e-4):
  The tolerance for the optimization based on the objective functional.
  The optimization code checks a termination criterion for optimality and continues until it is smaller than `tol`.
- `gamma` (float, default=1):
  The initial value of the continuation parameter $\gamma$ that will be used (only OrthogonallyWeightedL21Continuation).
- `gamma_tol` (float, default=1e-6):
  The smallest value of the continuation parameter $\gamma$ that will be used (only OrthogonallyWeightedL21Continuation).
- `normalize` (bool, default=False):
  If True, the regressors A will be normalized before regression by subtracting the mean and dividing by the l2-norm.
  Warning: this decreases the spark of A.
- `data_SVD_cutoff` (float, default=None):
  If set to a non-negative number, a singular value decomposition will be preformed
  and the data vector `Y` will be replaced with a full rank `Y_reduced` of dimension `(n_features, n_targets_reduced)`
  with `Y` approximating `Y_reduced @ Q` for some orthogonal `Q`
  with $|| Y_{reduced} Q - Y ||$ smaller than `data_SVD_cutoff`.
  The problem will be solved on the reduced representation with `n_targets_reduced`.
- `warm_start` (bool, default=False):
  When set to `True`, reuse the solution of the previous call to fit as initialization.
  Otherwise, erase the previous solution and perform random or zero initialization.
- `verbose` (int, default=False): Print information on updates.
  When set to `True`, prints at every step.
  When set to a integer `N`, prints every `N` steps.

## Attributes

- `coef_` (ndarray of shape (n_features, n_targets)):
  Parameter vector (Z in the cost function formula).
  If a 1D data array Y is passed in at fit (non multi-task usage), `coef_` is then a 1D array.

## Usage

```python
from base import OrthogonallyWeightedL21
import numpy as np

owl = OrthogonallyWeightedL21(noise_level=0.0001,
                              tol=1e-4,
                              max_iter=5000,
                              verbose=True)

A = np.array([[0.2, 1., 0., 0., 0.],
              [0.2, 0., 1., 0., 0.],
              [0.,  0., 0., 1., 0.],
              [0.,  0., 0., 0., 1.]])

X = np.array([[0., 0., 0., 1.,  1.], 
              [0., 0., 0., 1., -1.]]).transpose()
Y = A @ X

owl.fit(A, Y)

print('Z = ', owl.coef_)
print('singular values = ', np.linalg.svd(owl.coef_, full_matrices=False)[1])
print('fit = ', np.linalg.norm(A @ owl.coef_ - Y))
```
