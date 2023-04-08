import numpy as np


def _reweighted_row_inners(Z1, Z2, weight):
    """
    Computes the W-inner products between rows of Z1 and Z2
    """
    prod = np.expand_dims(np.sum((Z1 @ weight) * Z2, axis=1), axis=1)
    return prod


def _reweighted_row_norms(Z, weight='eye'):
    """
    Computes the W-norm of a matrix
    """
    return np.sqrt(_reweighted_row_inners(Z, Z, weight))


def _fidelity(A, Z, Y):
    return np.linalg.norm(A @ Z - Y)


def _regularizer(Z, weight):
    return np.sum(_reweighted_row_norms(Z, weight))


def _objective(A, Z, Y, weight, alpha):
    return (1 / 2) * _fidelity(A, Z, Y) ** 2 + alpha * _regularizer(Z, weight)


def _prox(Z, weight, eta):
    """
    Computes the W-proximal operator
    """
    norms = _reweighted_row_norms(Z, weight)
    prox = Z * np.maximum(0, 1 - eta / np.maximum(norms, eta / 2))
    return prox


def reweighted_coordinate_descent_multi_task(
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

    # initial step size
    initial_step_size = 1 / (np.linalg.norm(A.transpose() @ A, 2) * 2)
    step_size = initial_step_size

    # initial weights
    I = np.eye(n_tasks, n_tasks)
    weight_inv = Z.transpose() @ Z
    try:
        weight = np.linalg.inv(weight_inv)
    except:
        raise ValueError("Encountered a singular matrix. "
              "For improved results, consider using the OrthogonallyWeightedL21Continued algorithm.")

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
        weight_inv = Z.transpose() @ Z
        try:
            weight = np.linalg.inv(weight_inv)
        except:
            raise ValueError("Encountered a singular matrix. "
                             "For improved results, consider using the OrthogonallyWeightedL21Continued algorithm.")
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
                # update with new hyperparameters
                weight_inv = Z.transpose() @ Z
                try:
                    weight = np.linalg.inv(weight_inv)
                except:
                    raise ValueError("Encountered a singular matrix. "
                                     "For improved results, consider using the OrthogonallyWeightedL21Continued algorithm.")
                current_objective = _objective(A, Z, Y, weight, alpha)
                current_objective = _objective(A, Z, Y, weight, alpha)
            else:
                if (current_objective - old_objective) / pred > 3 / 4:
                    step_size = step_size * 1.5
                elif (current_objective - old_objective) / pred < 1 / 2:
                    step_size = step_size * (2 / 3)

        if verbose and k % 50 == 0:
            print('%6d: %3d' % (k, ((Z * Z).sum(axis=1) > 1.e-4).sum()), 'a=%1.2e' % (alpha),
                  'fit=%1.2e, obj=%1.2e, obj_err=%1.2e' % (fidelity, current_objective, -pred / step_size),
                  'stepsize=%1.1e' % (step_size / initial_step_size))

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

    # initial alpha
    if not alpha:
        gradZ = A.transpose() @ (A @ Z - Y)
        alpha = 0.5 * np.max(_reweighted_row_norms(gradZ, 'eye'))

    # initial weights
    I = np.eye(n_tasks, n_tasks)
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
