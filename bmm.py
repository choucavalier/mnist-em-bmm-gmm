import numpy as np

import mixture

class bmm(mixture.mixture):

    def __init__(self, n_components, n_iter=100, verbose=False):

        super().__init__(n_components, n_iter=n_iter, verbose=verbose)

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
