import datetime

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import mixture
import classifier

class bmm(mixture.mixture):

    def __init__(self, n_components, n_iter=100, tol=1e-3, verbose=False):

        super().__init__(n_components, n_iter, tol, verbose)

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

        n = x.shape[0]

        n_ms = np.sum(z, 0)
        # updating the means
        self.means = np.dot(np.diag(1 / n_ms), np.dot(z.T, x))
        # updating the weights
        self.weights = n_ms / n
