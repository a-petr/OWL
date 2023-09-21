import numpy as np
from scipy import linalg


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
    return np.linalg.norm(A @ Z - Y)

def _regularizer(Z, weight):
    """
    Computes the weighted l_{1,2} norm
    """
    return np.sum(_reweighted_row_norms(Z, weight))

def _obj(fidelity, regularizer, alpha):
    return (1 / (2 * alpha)) * fidelity ** 2 + regularizer

def _objective(A, Z, Y, weight, alpha):
    return _obj(_fidelity(A, Z, Y), _regularizer(Z, weight))


def check_discrepancy_principle(alpha, fidelity, noise_level):
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


def compute_weight(Z, gamma=0):
    
    n_tasks = Z.shape[1]
    
    R, p = linalg.qr(Z, mode='r', pivoting=True)
    R = R[0:n_tasks,:]
    RPt = R[:, np.argsort(p)]
    weight_inv = RPt.transpose() @ RPt
    PRinv = linalg.inv(R)[p, :]
    weight = PRinv @ PRinv.transpose()

    #weight_inv_alt = Z.transpose() @ Z
    #weight_alt = linalg.inv(weight_inv_alt)
    #print(weight_alt - weight)

    # indicator for condition number
    quot = np.abs(R[n_tasks-1,n_tasks-1]) / np.abs(R[0,0])
    
    return weight_inv, weight, quot


## why is this coordinate descent??
def reweighted_coordinate_descent_multi_task(
        Z,
        A,
        Y,
        max_iter,
        noise_level,
        tol,
        alpha=False,
        verbose=True):
    
    n_features = A.shape[1]
    n_tasks = Y.shape[1]
    
    # initial step size
    initial_step_size = 1 / (np.linalg.norm(A.transpose() @ A, 2) * 2)
    step_size = initial_step_size

    fidelity_at_last_alpha = False # internal variable for alpha adjustment
    
    # initial weights
    try:
        weight_inv, weight, _ = compute_weight(Z)
    except:
        raise ValueError("Singular weight matrix in initial step. "
              "Choose different initialization, or consider using the OrthogonallyWeightedL21Continued algorithm.")

    # initialize alpha if not specified
    if not alpha:
        gradZ = A.transpose() @ Y
        alpha = 0.5 * np.max(_reweighted_row_norms(gradZ, weight))
    
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
            weight_inv, weight, quot = compute_weight(Z)
            
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

            if pred >= 0 or step_size < 1e-10:
                ## error or just return what we have?
                print('failed to converge: pred=%1.2e, stepsize=%1.2e' % (pred, step_size))
                finished = True
        else:
            # proximal step is accepted

            reference_objective = _obj(max(fidelity, noise_level), regularizer, alpha);
            if obj_res < tol * reference_objective:
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

        if verbose and (k % 1 == 0 or finished):
            #print('%6d: %3d' % (k, ((Z * Z).sum(axis=1) > 1.e-4).sum()), 'a=%1.2e' % (alpha),
            #      'fit=%1.2e, obj=%1.2e, obj_err=%1.2e' % (fidelity, current_objective, -pred / step_size),
            #      'stepsize=%1.1e' % (step_size / initial_step_size))
            print('%6d: %3d' % (k, ((Z * Z).sum(axis=1) > 1.e-4).sum()), 'a=%1.2e' % (alpha),
                  'fit=%1.2e, reg=%1.2e, obj_err=%1.2e' % (fidelity, regularizer, -pred / step_size),
                  'stepsize=%1.1e' % (step_size / initial_step_size))
        if finished:
            break

    if verbose and quot <= 1e-3:
        print('condition number of Z too big %1.2e, stagnation likely' % (1/quot))
        
    return Z

def reweighted_coordinate_descent_multi_task_continued(
        Z,
        A,
        Y,
        max_iter,
        noise_level,
        tol,
        alpha=False,
        verbose=True,
        gamma_tol=1e-6):
    
    n_features = A.shape[1]
    n_tasks = Y.shape[1]

    # initial gamma
    gamma = 1

    # initial step size
    initial_step_size = 1 / (np.linalg.norm(A.transpose() @ A, 2) * 2)
    step_size = initial_step_size

    I = np.eye(n_tasks, n_tasks)

    # initial alpha
    if not alpha:
        gradZ = A.transpose() @ (A @ Z - Y)
        alpha = 0.5 * np.max(_reweighted_row_norms(gradZ, I))

    # initial weights
    weight_inv = (1 - gamma) * Z.transpose() @ Z + gamma * I
    weight = np.linalg.inv(weight_inv)
    current_objective = _objective(A, Z, Y, weight, alpha)

    for k in range(max_iter):
        Z_old = Z
        old_objective = current_objective
        weight_old = weight
        weight_inv_old = weight_inv

        # update Z
        Lam = np.zeros(weight.shape)
        reweighted_row_norms = _reweighted_row_norms(Z, weight)
        for f_iter in range(n_features):
            if reweighted_row_norms[f_iter] == 0:
                continue
            Lam += Z[f_iter:f_iter + 1, :].transpose() @ Z[f_iter:f_iter + 1, :] / reweighted_row_norms[f_iter]
        gradZ = (A.transpose() @ (A @ Z - Y)) @ weight_inv - (1 - gamma) * alpha * Z @ weight @ Lam
        Z = Z - step_size * gradZ
        Z = _prox(Z, weight, step_size * alpha)

        # update weights
        weight_inv = (1 - gamma) * Z.transpose() @ Z + gamma * I
        weight = np.linalg.inv(weight_inv)
        current_objective = _objective(A, Z, Y, weight, alpha)

        if np.isnan(current_objective) or np.isinf(current_objective) or current_objective > old_objective:
            step_size = step_size / 2
            Z = Z_old
            current_objective = old_objective
            weight = weight_old
            weight_inv = weight_inv_old
        else:
            pred = np.sum(_reweighted_row_inners(Z - Z_old, gradZ, weight)) + \
                   alpha * (np.sum(_reweighted_row_norms(Z, weight) - _reweighted_row_norms(Z_old, weight)))
            fidelity = _fidelity(A, Z, Y)
            if -pred / step_size < tol * (1 / 2) * max(fidelity, noise_level) ** 2:
                finished = False
                if fidelity > 1.5 * noise_level:
                    alpha = 0.9 * alpha
                elif fidelity < 0.75 * noise_level and gamma > gamma_tol:
                    # dot increase alpha if gamma \approx 0, since it does not always lead to a increase in fidel
                    alpha = 1.2 * alpha
                else:
                    finished = (gamma <= gamma_tol)
                    gamma = max(1e-6, gamma - 0.1)

                # update with new hyperparameters
                weight_inv = (1 - gamma) * Z.transpose() @ Z + gamma * I
                weight = np.linalg.inv(weight_inv)
                current_objective = _objective(A, Z, Y, weight, alpha)

                if finished:
                    break

            else:
                if (current_objective - old_objective) / pred > 3 / 4:
                    step_size = step_size * 1.5
                elif (current_objective - old_objective) / pred < 1 / 2:
                    step_size = step_size * (2 / 3)

        if verbose and k % 50 == 0:
            print('%6d: %3d' % (k, ((Z * Z).sum(axis=1) > 1.e-4).sum()), 'a=%1.2e, g=%1.2e' % (alpha, gamma),
                  'fit=%1.2e, obj=%1.2e, obj_err=%1.2e' % (fidelity, current_objective, -pred / step_size),
                  'stepsize=%1.1e' % (step_size / initial_step_size))

    return Z
