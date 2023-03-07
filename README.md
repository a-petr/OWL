# (OWL) Orthogonally Weighted L_{21}  regularizer

Multi-task Lasso model trained with Orthogonally Weighted L_{2,1} (OWL) regularizer. The model optimizes the following objective function:

$$\frac{1}{2n} ||Y - AX||\_{\text{Fro}}^2 + \alpha  ||W(W^TW)^{-1}||\_{2,1} $$

where $||W||\_{2,1}$ is defined as the sum of norms of each row: $\sum_i \sqrt{\sum\_j W\_{ij}^2}$. 


For more information on this model, please refer to "Orthogonally weighted $L_{2,1}$  regularization for rank-aware joint sparse recovery: algorithm and analysis" by A. Petrosyan, K. Pieper, H. Tran.

## Parameters

- `alpha` (float, default=False): Constant that multiplies the regularizing term. Defaults to False in which case an analytic formula is used.
- `normalize` (bool, default=False): If True, the regressors A will be normalized before regression by subtracting the mean and dividing by the l2-norm. Warning: this decreases the spark of A.         
- `max_iter` (int, default=1000): The maximum number of iterations.
- `noise_level` (float, default=1.0): Estimated noise level.
- `tol` (float, default=1e-4): The tolerance for the optimization: if the updates are smaller than `tol`, the optimization code checks the dual gap for optimality and continues until it is smaller than `tol`.
- `warm_start` (bool, default=False): When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
- `verbose` (bool, default=False): When set to True, prints at every 50 steps.
- `gamma_tol` (float, default=10e-4): Sets a limit on how small gamma can get.
__
## Attributes

- `coef_` (ndarray of shape (n_targets, n_features)): Parameter vector (W in the cost function formula). If a 1D y is passed in at fit (non multi-task usage), `coef_` is then a 1D array. Note that `coef_` stores the transpose of `X`, `X.T`.

## Usage

```python
from source.basealgorithm import OrthogonallyWeightedL21

owl = OrthogonallyWeightedL21(alpha=0.1,
                              noise_level=0.0001,
                              normalize=False,
                              tol=1e-2,
                              max_iter=50000,
                              verbose=True)A = np.array([[0., 1., 2.], [3., 4., 5.]])

Y = np.array([[-1.,  1.], [-1.,  4.]])
owl.fit(A, Y)
print(clf.coef_)
```
