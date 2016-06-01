import numpy as np

import mixture

EPS = np.finfo(float).eps

def _log_multivariate_normal_density_diag(x, means, covars):
    '''Compute Gaussian log-density at x for a diagonal model.'''
    d = x.shape[1]
    lpr = -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(x, (means / covars).T)
                  + np.dot(x ** 2, (1.0 / covars).T))
    return lpr

class gmm(mixture.mixture):

    def __init__(self, n_components, verbose=False):

        super().__init__(n_components, init_params='wmc', verbose=verbose)

    def _log_support(self, x):

        lpr = _log_multivariate_normal_density_diag(x, self.means, self.covars)

        return lpr

    def _do_mstep(self, x, z):

        weights = z.sum(axis=0)
        weighted_x_sum = np.dot(z.T, x)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        self.weights = (weights / (weights.sum() + 10 * EPS) + EPS)
        self.means = weighted_x_sum * inverse_weights

        self.covars = self._covar_mstep_diag(x, z, weighted_x_sum,
                                             inverse_weights)

    def _covar_mstep_diag(self, x, responsibilities, weighted_x_sum, norm):
        '''Perform the covariance M step for diagonal cases.'''
        avg_x2 = np.dot(responsibilities.T, x * x) * norm
        avg_means2 = self.means ** 2
        avg_x_means = self.means * weighted_x_sum * norm
        return avg_x2 - 2 * avg_x_means + avg_means2 + self.min_covar
