# Authors: Armenak Petrosyan <apetrosyan3@gatech.edu>
#          Konstantin Pieper <pieperk@ornl.gov>
#          Hoang Tran <tranh@ornl.gov>


import numpy as np
import utils as ut


class OrthogonallyWeightedL21:
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

        return self


if __name__ == "__main__":
    owl = OrthogonallyWeightedL21(alpha=0.1,
                                  noise_level=0.0001,
                                  normalize=False,
                                  tol=1e-2,
                                  max_iter=50000,
                                  verbose=True)
    A = np.array([[1., 1., 0], [1., 0., 1.]])
    Y = np.array([[2., 0.], [1., 1.]])
    X = np.array([[1., 1.], [1., -1.], [0., 0.]])
    owl.fit(A, Y)
    print(owl.coef_)
    print(np.linalg.norm(A @ owl.coef_ - Y))
