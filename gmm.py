import numpy as np

def _log_multivariate_normal_density_diag(x, means, covars):
    '''Compute Gaussian log-density at x for a diagonal model.'''
    n_samples, n_dim = x.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(x, (means / covars).T)
                  + np.dot(x ** 2, (1.0 / covars).T))
    return lpr
