# Authors: Armenak Petrosyan <apetrosyan3@gatech.edu>
#          Konstantin Pieper <pieperk@ornl.gov>
#          Hoang Tran <tranh@ornl.gov>


import numpy as np
import utils as ut


class OrthogonallyWeightedL21:
    def __init__(
            self,
            alpha=None,
            noise_level=None,
            max_iter=5000,
            tol=1e-4,
            warm_start=False,
            normalize=False,
            verbose=False
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize = normalize
        self.tol = tol
        self.warm_start = warm_start
        self.noise_level = noise_level
        self.verbose = verbose
        self.coef_ = None

    def fit(self, A, Y):
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
            #self.coef_ = np.zeros((n_features, n_targets), dtype=A.dtype.type)
            # random intialization
            self.coef_ = np.random.randn(n_features, n_targets)
            # ensure the random matrix is well conditioned
            #self.coef_ = np.linalg.svd(owl.coef_, full_matrices=False)[0]


        self.coef_ = ut.reweighted_coordinate_descent_multi_task(
            self.coef_,
            A,
            Y,
            self.alpha,
            self.noise_level,
            self.max_iter,
            self.tol,
            self.verbose)

        return self


class OrthogonallyWeightedL21Continuation:
    def __init__(
            self,
            alpha=None,
            noise_level=None,
            max_iter=1000,
            tol=1e-4,
            gamma_tol=1e-6,
            warm_start=False,
            normalize=False,
            verbose=False
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize = normalize
        self.tol = tol
        self.gamma_tol = gamma_tol
        self.warm_start = warm_start
        self.noise_level = noise_level
        self.verbose = verbose
        self.coef_ = None

    def fit(self, A, Y):
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
            # random intialization
            #self.coef_ = np.random.randn(n_features, n_targets)

        self.coef_ = ut.reweighted_coordinate_descent_multi_task_continuation(
            self.coef_,
            A,
            Y,
            self.alpha,
            self.noise_level,
            self.max_iter,
            self.tol,
            self.gamma_tol,
            self.verbose)

        return self


if __name__ == "__main__":
    owl = OrthogonallyWeightedL21(noise_level=0.0001,
                                  tol=1e-4,
                                  max_iter=5000,
                                  verbose=True)
    A = np.array([[1., 1., 0], [1., 0., 1.]])
    Y = np.array([[2., 0.], [1., 1.]])
    ## this problem has many solutions
    X1 = np.array([[1., 1.], [1., -1.], [0., 0.]])
    X2 = np.array([[0., 0.], [2., 0.], [1., 1.]])
    X3 = np.array([[2., 0.], [0., 0.], [-1., 1.]])
    owl.fit(A, Y)
    print('Z = ', owl.coef_)
    print('singular values = ', np.linalg.svd(owl.coef_, full_matrices=False)[1])
    print('fit = ', np.linalg.norm(A @ owl.coef_ - Y))
