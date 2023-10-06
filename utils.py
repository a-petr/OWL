import numpy as np
from scipy import linalg


def normalize_matrix_and_data(A, Y):
    """
    Normalize the matrix A and data vector Y
    """
    A_mean = A.mean(axis=0)
    A_std = A.std(axis=0)
    A_std[A_std == 0] = A_mean[A_std == 0]
    Y_mean = Y.mean(axis=0)

    ## this overwrites the memory of A and Y (use copy()?)
    A -= A_mean[None, :]
    A /= A_std[None, :]
    Y -= Y_mean[None, :]
    
    return A, Y

def data_SVD_preprocess(Y, cutoff, verbose=False):
    """
    Use the singular value decomposition to find a decomposition
    
    Y = Y_reduced @ Q

    Y_reduced has dimension n_features, n_targets_reduced and singular values bigger than cutoff
    Q has dimension n_targets_reduced, n_targets
    """

    U, s, V = linalg.svd(Y, full_matrices=False)

    tail_error = np.flip(np.sqrt(np.cumsum(np.flip(s ** 2))))

    if np.any(tail_error <= cutoff):
        Y_reduced = U[:, tail_error > cutoff] * s[tail_error > cutoff]
        Q = V[tail_error > cutoff, :]

        if verbose:
            print('reduced data size from %d to %d with error %1.2f' % \
                  (Y.shape[1], Y_reduced.shape[1], np.max(tail_error[tail_error <= cutoff])))
    else:
        Y_reduced = Y
        Q = np.eye(Y.shape[1])

    return Y_reduced, Q


def _reweighted_row_inners(Z1, Z2, weight):
    """
    Computes the W-inner products between rows of Z1 and Z2
    """
    prod = np.expand_dims(np.sum((Z1 @ weight) * Z2, axis=1), axis=1)
    return prod

def _reweighted_row_norms(Z, weight):
    """
    Computes the W-norm of a matrix
    """
    return np.sqrt(_reweighted_row_inners(Z, Z, weight))

def _prox(Z, weight, eta):
    """
    Computes the W-proximal operator
    """
    norms = _reweighted_row_norms(Z, weight)
    prox = Z * np.maximum(0, 1 - eta / np.maximum(norms, eta / 2))
    return prox


def _fidelity(A, Z, Y):
    """
    Computes the fidelity or fit of the data
    """
    return np.linalg.norm(A @ Z - Y)

def _regularizer(Z, weight):
    """
    Computes the weighted l_{1,2} norm
    """
    return np.sum(_reweighted_row_norms(Z, weight))

def _obj(fidelity, regularizer, alpha):
    """
    Computes the objective function for minimization
    """
    return (1 / (2 * alpha)) * fidelity ** 2 + regularizer

def _objective(A, Z, Y, weight, alpha):
    return _obj(_fidelity(A, Z, Y), _regularizer(Z, weight))


def _compute_weight(Z):
    """
    compute the weight matrix in a way that guarantees symmetry for weight
    and gives an estimate of the reciprocal condition number

    weight_inv = Z.transpose() @ Z
    weight = linalg.inv(weight_inv)
    """
    
    n_targets = Z.shape[1]
    
    R, p = linalg.qr(Z, mode='r', pivoting=True)
    R = R[0:n_targets,:]
    RPt = R[:, np.argsort(p)]
    weight_inv = RPt.transpose() @ RPt
    PRinv = linalg.inv(R)[p, :]
    weight = PRinv @ PRinv.transpose()

    # indicator for condition number
    rcond = np.abs(R[n_targets-1,n_targets-1]) / np.abs(R[0,0])
    
    return weight_inv, weight, rcond


def check_discrepancy_principle(alpha, fidelity, noise_level):
    """
    Check if the discrepancy principle is fulfilled, and suggest a new value for alpha
    """
    kappa1 = 0.75
    kappa2 = 1.5
    if fidelity > kappa2 * noise_level:
        # decrease alpha to improve fit
        alpha_new = 0.9 * alpha
    elif fidelity < kappa1 * noise_level:
        # increase alpha to improve regularization
        alpha_new = 1.2 * alpha
    else:
        # discrepancy principle fulfilled
        alpha_new = alpha

    return alpha_new


def reweighted_l21_multi_task(
        Z,
        A,
        Y,
        alpha,
        noise_level,
        max_iter,
        tol,
        verbose=True):
    
    n_features = A.shape[1]
    n_targets = Y.shape[1]

    if noise_level:
        fidelity_at_last_alpha = False # internal variable for alpha adjustment
    
    # initial weights
    try:
        weight_inv, weight, _ = _compute_weight(Z)
    except:
        raise ValueError("Singular weight matrix in initial step. "
              "Choose different initialization, or consider using the OrthogonallyWeightedL21Continuation algorithm.")

    # initialize alpha if not specified
    if not alpha:
        gradZ = A.transpose() @ Y
        alpha = 0.1 * np.max(_reweighted_row_norms(gradZ, weight))

    # initial step size
    reference_step_size = alpha / (np.linalg.norm(A, 2) ** 2)
    step_size = reference_step_size

    fidelity = _fidelity(A, Z, Y)
    regularizer = _regularizer(Z, weight)
    current_objective = _obj(fidelity, regularizer, alpha)

    for k in range(max_iter):
        Z_old = Z
        old_objective = current_objective
        weight_old = weight
        weight_inv_old = weight_inv

        # update Z
        Lam = np.zeros(weight.shape)
        reweighted_row_norms = _reweighted_row_norms(Z, weight)

        nonzero_rows = np.nonzero(np.squeeze(reweighted_row_norms))[0]
        for f_iter in nonzero_rows:
            Lam += Z[f_iter:f_iter + 1, :].transpose() @ Z[f_iter:f_iter + 1, :] / reweighted_row_norms[f_iter]

        gradZ = (1/alpha) * (A.transpose() @ (A @ Z - Y)) @ weight_inv - Z @ weight @ Lam
        Z = Z - step_size * gradZ
        Z = _prox(Z, weight, step_size)

        # compute the predicted decrease
        pred = np.sum(_reweighted_row_inners(Z - Z_old, gradZ, weight)) + \
            np.sum(_reweighted_row_norms(Z, weight) - _reweighted_row_norms(Z_old, weight))

        # functional residual
        obj_res = -pred / step_size

        # update weights and objective if weight exists
        try:
            weight_inv, weight, rcond = _compute_weight(Z)
            
            fidelity = _fidelity(A, Z, Y)
            regularizer = _regularizer(Z, weight)
            current_objective = _obj(fidelity, regularizer, alpha)
            failed_update = np.isnan(current_objective) or np.isinf(current_objective)
        except:
            weight = weight_old
            failed_update = True
            
        finished = False        
        kappa = 0.001
        if failed_update or current_objective - old_objective > kappa * pred:
            # discard current proximal step (backtracking line-search)
            step_size = step_size / 2
            Z = Z_old
            current_objective = old_objective
            weight = weight_old
            weight_inv = weight_inv_old

            if pred >= 0 or step_size * rcond < 1e-10:
                ## error or just return what we have?
                print('Failed to converge: pred=%1.2e, stepsize=%1.2e, condest=%1.2e.' % (pred, step_size, 1/rcond))
                print('Consider using the OrthogonallyWeightedL21Continuation algorithm.')
                finished = True
        else:
            # proximal step is accepted

            reference_objective = _obj(fidelity, regularizer, alpha);
            if not noise_level and obj_res < tol * reference_objective:
                # termination criterion is fulfilled
                finished = True
            elif obj_res < tol * reference_objective:
                # termination criterion for the proximal gradient method is fulfilled
                # check if we need to update alpha to fulfill the discrepancy principle
                alpha_new = check_discrepancy_principle(alpha, fidelity, noise_level)

                if alpha_new > alpha:
                    if fidelity_at_last_alpha and fidelity <= fidelity_at_last_alpha:
                        # fidelity did not go up, increasing alpha does not help anymore
                        finished = True
                    else:
                        fidelity_at_last_alpha = fidelity
                elif alpha_new < alpha:
                    fidelity_at_last_alpha = False
                else:
                    # discrepancy principle fulfilled
                    finished = True

                reference_step_size *= alpha_new / alpha
                alpha = alpha_new
                # update with new hyperparameter alpha for next proximal update
                current_objective = _obj(fidelity, regularizer, alpha)
            else:
                # continue iterating the proximal gradient
                # Performance optimization: try to get a better guess for the next stepsize
                #  depending on the agreement of functional and model
                if (current_objective - old_objective) / pred > 3 / 4:
                    step_size = step_size * 1.5
                elif (current_objective - old_objective) / pred < 1 / 3:
                    step_size = step_size * (2 / 3)

        if verbose and (k % verbose == 0 or finished):
            print('%6d: %3d' % (k, ((Z * Z).sum(axis=1) > 1.e-4).sum()),
                  'alpha=%1.1e' % (alpha),
                  'fit=%1.2e reg=%1.2f obj_err=%1.2e' % (fidelity, regularizer, obj_res),
                  'step=%1.1e' % (step_size / reference_step_size))
        if finished:
            break

    if verbose and rcond <= 1e-3:
        print('Estimated condition number of Z: %1.2e, stagnation likely' % (1/rcond))
        print('Consider different initialization or using the OrthogonallyWeightedL21Continuation algorithm.')
        
    return Z


def _compute_weight_gamma(Z, gamma):
    """
    compute the weight matrix in a way that guarantees symmetry for weight
    and gives an estimate of the reciprocal condition number

    weight_inv = gamma * I + (1-gamma) * Z.transpose() @ Z
    weight = linalg.inv(weight_inv)
    """
    
    n_targets = Z.shape[1]
    I = np.eye(n_targets, n_targets)
    weight_inv = gamma * I + (1-gamma) * Z.transpose() @ Z

    L = linalg.cholesky(weight_inv)
    Linv = linalg.inv(L)
    weight = Linv @ Linv.transpose()
    
    # indicator for condition number
    rcond = (np.min(np.diag(L)) / np.max(np.diag(L))) ** 2

    return weight_inv, weight, rcond


def _gamma_continuation_schedule(gamma, gamma_tol):
    """
    provide the next continuation parameter
    """
    return max(gamma_tol, gamma - 0.1) #gamma / 1.5


def reweighted_l21_multi_task_continuation(
        Z,
        A,
        Y,
        alpha,
        noise_level,
        max_iter,
        tol,
        gamma_0=1,
        gamma_tol=1e-6,
        verbose=True):
    
    n_features = A.shape[1]
    n_targets = Y.shape[1]

    # initial gamma
    gamma = gamma_0
    
    # initial weights
    weight_inv, weight, _ = _compute_weight_gamma(Z, gamma)

    # initialize alpha if not specified
    if not alpha:
        gradZ = A.transpose() @ Y
        alpha = 0.1 * np.max(_reweighted_row_norms(gradZ, weight))

    # initial step size
    reference_step_size = alpha / (np.linalg.norm(A, 2) ** 2)
    step_size = reference_step_size
    
    fidelity = _fidelity(A, Z, Y)
    regularizer = _regularizer(Z, weight)
    current_objective = _obj(fidelity, regularizer, alpha)

    for k in range(max_iter):
        Z_old = Z
        old_objective = current_objective
        weight_old = weight
        weight_inv_old = weight_inv

        # update Z
        Lam = np.zeros(weight.shape)
        reweighted_row_norms = _reweighted_row_norms(Z, weight)

        nonzero_rows = np.nonzero(np.squeeze(reweighted_row_norms))[0]
        for f_iter in nonzero_rows:
            Lam += (1 - gamma) * Z[f_iter:f_iter + 1, :].transpose() @ Z[f_iter:f_iter + 1, :] / reweighted_row_norms[f_iter]

        gradZ = (1/alpha) * (A.transpose() @ (A @ Z - Y)) @ weight_inv - Z @ weight @ Lam
        Z = Z - step_size * gradZ
        Z = _prox(Z, weight, step_size)

        # compute the predicted decrease
        pred = np.sum(_reweighted_row_inners(Z - Z_old, gradZ, weight)) + \
            np.sum(_reweighted_row_norms(Z, weight) - _reweighted_row_norms(Z_old, weight))

        # functional residual
        obj_res = -pred / step_size

        # update weights and objective
        weight_inv, weight, rcond = _compute_weight_gamma(Z, gamma)

        fidelity = _fidelity(A, Z, Y)
        regularizer = _regularizer(Z, weight)
        current_objective = _obj(fidelity, regularizer, alpha)
        failed_update = np.isnan(current_objective) or np.isinf(current_objective)

        # decide if to accept step or to update hyperparameters
        finished = False
        kappa = 0.001
        if failed_update or current_objective - old_objective > kappa * pred:
            # discard current proximal step (backtracking line-search)
            step_size = step_size / 2
            Z = Z_old
            current_objective = old_objective
            weight = weight_old
            weight_inv = weight_inv_old

            if pred >= 0 or step_size * rcond < 1e-10:
                print('Failed to converge: pred=%1.2e, stepsize=%1.2e, condest=%1.2e.' % (pred, step_size, 1/rcond))
                finished = True
        else:
            # proximal step is accepted
            gamma_update = False

            reference_objective = _obj(fidelity, regularizer, alpha);
            if not noise_level and obj_res < tol * reference_objective:
                # termination criterion is fulfilled
                # check if we need to decrease gamma
                if gamma <= gamma_tol:
                    finished = True
                else:
                    gamma = _gamma_continuation_schedule(gamma, gamma_tol)
                    gamma_update = True
            elif obj_res < tol * reference_objective:
                # termination criterion for the proximal gradient method is fulfilled
                # check if we need to update alpha to fulfill the discrepancy principle
                alpha_new = check_discrepancy_principle(alpha, fidelity, noise_level)
                if alpha == alpha_new or gamma <= gamma_tol:
                    # discrepancy principle fulfilled
                    # check if we need to decrease gamma
                    if gamma <= gamma_tol:
                        finished = True
                    else:
                        gamma = _gamma_continuation_schedule(gamma, gamma_tol)
                        gamma_update = True
                else: 

                    reference_step_size *= alpha_new / alpha
                    alpha = alpha_new
                    # update with new hyperparameter alpha for next proximal update
                    current_objective = _obj(fidelity, regularizer, alpha)
            else:
                # continue iterating the proximal gradient
                # Performance optimization: try to get a better guess for the next stepsize
                #  depending on the agreement of functional and model
                if (current_objective - old_objective) / pred > 3 / 4:
                    step_size = step_size * 1.5
                elif (current_objective - old_objective) / pred < 1 / 3:
                    step_size = step_size * (2 / 3)

            if gamma_update: 
                # gamma was updated: update weights and objective
                weight_inv, weight, rcond = _compute_weight_gamma(Z, gamma)

                regularizer = _regularizer(Z, weight)
                current_objective = _obj(fidelity, regularizer, alpha)

        if verbose and (k % verbose == 0 or finished):
            print('%6d: %3d' % (k, ((Z * Z).sum(axis=1) > 1.e-4).sum()),
                  'alpha=%1.1e gamma=%1.1e' % (alpha, gamma),
                  'fit=%1.2e reg=%1.2f obj_err=%1.2e' % (fidelity, regularizer, obj_res),
                  'step=%1.1e' % (step_size / reference_step_size))

        if finished:
            break
        
    return Z



