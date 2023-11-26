# Authors: Armenak Petrosyan <arm.petros@gmail.com>
#          Konstantin Pieper <pieperk@ornl.gov>
#          Hoang Tran <tranha@ornl.gov>


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
            data_SVD_cutoff=None,
            verbose=False
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize = normalize
        self.data_SVD_cutoff = data_SVD_cutoff
        self.tol = tol
        self.warm_start = warm_start
        self.noise_level = noise_level
        self.verbose = verbose
        self.coef_ = None

    def fit(self, A, Y):
        if self.normalize:
            A, Y = ut.normalize_matrix_and_data(A, Y)

        n_samples, n_features = A.shape
        n_targets = Y.shape[1]

        if self.data_SVD_cutoff is not None:
            Y, Q = ut.data_SVD_preprocess(Y, self.data_SVD_cutoff, self.verbose)
            n_targets = Y.shape[1]

        if not self.warm_start:
            # self.coef_ = np.zeros((n_features, n_targets), dtype=A.dtype.type)
            # random intialization
            self.coef_ = np.random.randn(n_features, n_targets)

        self.coef_ = ut.reweighted_l21_multi_task(
            self.coef_,
            A,
            Y,
            self.alpha,
            self.noise_level,
            self.max_iter,
            self.tol,
            self.verbose)

        if self.data_SVD_cutoff is not None:
            self.coef_ = self.coef_ @ Q

        return self


class OrthogonallyWeightedL21Continuation:
    def __init__(
            self,
            alpha=None,
            noise_level=None,
            max_iter=1000,
            tol=1e-4,
            gamma=1,
            gamma_tol=1e-6,
            warm_start=False,
            normalize=False,
            data_SVD_cutoff=None,
            verbose=False
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize = normalize
        self.data_SVD_cutoff = data_SVD_cutoff
        self.tol = tol
        self.gamma = gamma
        self.gamma_tol = gamma_tol
        self.warm_start = warm_start
        self.noise_level = noise_level
        self.verbose = verbose
        self.coef_ = None

    def fit(self, A, Y):
        if self.normalize:
            A, Y = ut.normalize_matrix_and_data(A, Y)

        n_samples, n_features = A.shape
        n_targets = Y.shape[1]

        if self.data_SVD_cutoff is not None:
            Y, Q = ut.data_SVD_preprocess(Y, self.data_SVD_cutoff, self.verbose)
            n_targets = Y.shape[1]

        if not self.warm_start:
            self.coef_ = np.zeros((n_features, n_targets), dtype=A.dtype.type)

        self.coef_ = ut.reweighted_l21_multi_task_continuation(
            self.coef_,
            A,
            Y,
            self.alpha,
            self.noise_level,
            self.max_iter,
            self.tol,
            self.gamma,
            self.gamma_tol,
            self.verbose)

        if self.data_SVD_cutoff is not None:
            self.coef_ = self.coef_ @ Q

        return self


if __name__ == "__main__":
    owl = OrthogonallyWeightedL21(noise_level=0.0001,
                                  tol=1e-4,
                                  max_iter=5000,
                                  verbose=True)

    A = np.array([[0.2, 1., 0., 0., 0.],
                  [0.2, 0., 1., 0., 0.],
                  [0.,  0., 0., 1., 0.],
                  [0.,  0., 0., 0., 1.]])

    X = np.array([[0., 0., 0., 1., 1.],
                  [0., 0., 0., 1., -1.]]).transpose()
    Y = A @ X

    owl.fit(A, Y)

    print('Z = ', owl.coef_)
    print('singular values = ', np.linalg.svd(owl.coef_, full_matrices=False)[1])
    print('fit = ', np.linalg.norm(A @ owl.coef_ - Y))
