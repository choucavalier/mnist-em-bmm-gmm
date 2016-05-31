import numpy as np

import mixture

EPS = np.finfo(float).eps

class bmm(mixture.mixture):

    def __init__(self, n_components, verbose=False):

        super().__init__(n_components, verbose=verbose)

    def _log_support(self, x):

        k = self.n_components; pi = self.weights; mu = self.means

        x_c = 1 - x
        mu_c = 1 - mu

        log_support = np.ndarray(shape=(x.shape[0], k))

        for i in range(k):
            log_support[:, i] = (
                np.sum(x * np.log(mu[i, :].clip(min=1e-50)), 1) \
                + np.sum(x_c * np.log(mu_c[i, :].clip(min=1e-50)), 1))

        return log_support

    def _do_mstep(self, x, z):

        weights = z.sum(axis=0)
        weighted_x_sum = np.dot(z.T, x)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        self.weights = (weights / (weights.sum() + 10 * EPS) + EPS)
        self.means = weighted_x_sum * inverse_weights
