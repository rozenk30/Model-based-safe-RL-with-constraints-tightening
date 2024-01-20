from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.spatial import distance

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.

    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)

    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [1]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=100):
    '''
    Proposes the next sampling point by optimizing the acquisition function.

    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 0.05
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        solution = -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)
        return solution.ravel()

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[0, 0], bounds[0, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x.reshape(-1, 1)

def get_Y(X_next, X_data_scaled):
    '''
    Get the next sampling Y point from the case study data.

    Args:
        X_next: Next sample X (scaled)
        X_data_scaled: All case study X data (scaled)

    Returns:
        min_idx: Sample idx.
    '''
    dist_gat = []
    for k in range(len(X_data_scaled)):
        X_k = X_data_scaled[k]
        dist = distance.euclidean(X_k, X_next)
        dist_gat.append(dist)

    min_dist = min(dist_gat)
    min_idx = np.argmin(dist_gat)
    print(min_idx, min_dist)

    return min_idx


def scale(data, max, min):
    data_scaled = (data - min) / (max - min)
    return data_scaled

def descale(data, max, min):
    data_descaled = (max - min) * data + min
    return data_descaled