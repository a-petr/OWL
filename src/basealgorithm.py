# Author: Armenak Petrosyan <apetrosyan3@gatech.edu>
#         Konstantin Pieper <pieperk@ornl.gov>
#         Hoang Tran <tranh@ornl.gov>



import numpy as np
import utils as ut


class OrthogonallyWeightedLasso():
    """Multi-task model trained with OWL-21 regularizer.
    The optimization objective for MultiTaskElasticNet is::
        (1 / (2 * n_samples)) * ||Y - AX||_Fro^2
        + alpha * l1_ratio * ||W(W^TW)^{-1}||_21
    Where::
        ||W||_21 = sum_i sqrt(sum_j W_ij ^ 2)
    i.e. the sum of norms of each row.
    Read more in the : A.Petrosyan, K.Pieper, H. Tran,
    "OWL: a rank aware regularization method for joint sparse recovery"
    Parameters
    ----------
    alpha : float, default=False
        Constant that multiplies the regularizing term. Defaults to False in which case an analytic formula is used.
    normalize : bool, default=False
        If True, the regressors A will be normalized before regression by
        subtracting the mean and dividing by the l2-norm.
        Warning: this decreases the spark of A
    max_iter : int, default=1000
        The maximum number of iterations.
    noise_level : float, default=1.0
        estimated noise level
    tol : float, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
    verbose : bool, default=False
        When set to ``True``, prints at every 50 steps.
    gamma_tol : float, default=10e-4
        Sets a limit on how small gamma can get.
    Attributes
    ----------
    coef_ : ndarray of shape (n_targets, n_features)
        Parameter vector (W in the cost function formula). If a 1D y is
        passed in at fit (non multi-task usage), ``coef_`` is then a 1D array.
        Note that ``coef_`` stores the transpose of ``X``, ``X.T``.
    n_iter_ : int
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    eps_ : float
        The tolerance scaled by the variance of the target `y`.
    sparse_coef_ : sparse matrix of shape (n_features,) or \
            (n_targets, n_features)
        Sparse representation of the `coef_`.
    n_features_in_ : int
        Number of features seen during :term:`fit`.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    Examples
    --------
    clf = OrthogonallyWeightedLasso(alpha=0.1, noise_level=0, normalize=True)
    A = np.array([[0., 1., 2.], [3., 4., 5.]])
    Y = np.array([[-1.,  1.], [-1.,  4.]])
    clf.fit(A, Y)
    print(clf.coef_)
    [[ 0.49543403  0.49854937]
     [ 0.          0.        ]
     [-0.49751728  0.50064646]]
    """

    def __init__(
            self,
            alpha=False,
            normalize=False,
            max_iter=1000,
            noise_level=1,
            tol=1e-4,
            warm_start=False,
            verbose=False,
            gamma_tol=False
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize = normalize
        self.tol = tol
        self.warm_start = warm_start
        self.noise_level = noise_level
        self.gamma_tol = gamma_tol
        self.verbose = verbose

    def fit(self, A, Y):
        """Fit MultiTaskElasticNet model with coordinate descent.
        Parameters
        ----------
        A : ndarray of shape (n_samples, n_features)
            Data.
        y : ndarray of shape (n_samples, n_targets)
            Target. Will be cast to A's dtype if necessary.
        Returns
        -------
        self : object
            Fitted estimator.
        Notes
        -----
        Coordinate descent is an algorithm that considers each column of
        data at a time hence it will automatically convert the X input
        as a Fortran-contiguous numpy array if necessary.
        """
        # normalize each feature in the sensing matrix A by (A - mean(A)) / std(A)
        # corresponds to Y - mean(Y) dividing final solution by std(A) for the data matrix
        if self.normalize:
            A_mean = A.mean(axis=0)
            A_std = A.std(axis=0)
            A_std[A_std == 0] = A_mean[A_std == 0]
            Y_mean = Y.mean(axis=0)

            A -= A_mean[None, :]
            A /= A_std[None, :]
            Y -= Y_mean[None, :]

        n_samples, n_features = A.shape
        n_targets = Y.shape[1]

        if not self.warm_start:
            self.coef_ = np.zeros((n_features, n_targets), dtype=A.dtype.type)

        self.coef_ = ut.reweighted_coordinate_descent_multi_task(
            self.coef_,
            A,
            Y,
            self.max_iter,
            self.noise_level,
            self.tol,
            self.alpha,
            self.verbose,
            self.gamma_tol)

        # if self.normalize:
        #     self.coef_ /= A_std[:, None]

        # return self for chaining fit and predict calls
        return self


if __name__ == "__main__":
    clf = OrthogonallyWeightedLasso(alpha=0.1, noise_level=0.0001, normalize=False, tol=1e-2, max_iter=50000, verbose=True)
    A = np.array([[1., 1., 0], [1., 0., 1.]])
    Y = np.array([[2.,  0.], [1.,  1.]])
    X = np.array([[1., 1.], [1., -1.], [0., 0.]])
    clf.fit(A, Y)
    print(clf.coef_)
    print(np.linalg.norm(A@clf.coef_ - Y))