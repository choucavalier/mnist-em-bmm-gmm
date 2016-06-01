from datetime import datetime
import sys

import numpy as np
from sklearn.cluster import KMeans

EPS = np.finfo(float).eps

class mixture:

    def __init__(self, n_components, init_params='wm', n_iter=100,
                 tol=1e-3, min_covar=1e-4, verbose=False):

        #: number of components in the mixture
        self.n_components = n_components
        #: params to init
        self.init_params = init_params
        #: max number of iterations
        self.n_iter = n_iter
        #: convergence threshold
        self.tol = tol
        self.min_covar = min_covar
        self.verbose = verbose

        k = self.n_components

        self.weights = np.array([1 / k for _ in range(k)])
        self.means = None
        self.covars = None

        self.converged_ = False

    def fit(self, x, means_init_heuristic='random', means=None, labels=None):

        k = self.n_components
        n = x.shape[0]
        d = x.shape[1]

        self.means = np.ndarray(shape=(k, d))

        # initialization of the means
        if 'm' in self.init_params:
            if self.verbose:
                print('using {} heuristic to initialize the means'
                      .format(means_init_heuristic))
            if means_init_heuristic == 'random':
                self.means = np.random.rand(k, d) * 0.5 + 0.25
            elif means_init_heuristic == 'data_classes_mean':
                if labels is None:
                    raise ValueError(
                        'labels required for data_classes_mean init')
                self.means = _data_classes_mean_init(x, labels)
            elif means_init_heuristic == 'kmeans':
                self.means = _kmeans_init(x, k, means=means,
                                          verbose=self.verbose)

        # initialization of the covars
        if 'c' in self.init_params:
            if self.verbose:
                print('initializing covars')
            cv = np.cov(x.T) + self.min_covar * np.eye(x.shape[1])
            self.covars = np.tile(np.diag(cv), (k, 1))

        start = datetime.now()

        iterations = 0

        prev_log_likelihood = None
        current_log_likelihood = -np.inf

        while iterations <= self.n_iter:

            elapsed = datetime.now() - start

            prev_log_likelihood = current_log_likelihood

            # expectation step
            log_likelihoods, responsibilities = self.score_samples(x)
            current_log_likelihood = log_likelihoods.mean()

            if self.verbose:

                print('[{:02d}] likelihood = {} (elapsed {})'
                      .format(iterations, current_log_likelihood, elapsed))

            if prev_log_likelihood is not None:
                change = abs(current_log_likelihood - prev_log_likelihood)
                if change < self.tol:
                    self.converged_ = True
                    break

            self._do_mstep(x, responsibilities)

            iterations += 1

        end = datetime.now()

        elapsed = end - start

        print('converged in {} iterations in {}'
              .format(iterations, elapsed))

    def _do_mstep(self, x, z):

        weights = z.sum(axis=0)
        weighted_x_sum = np.dot(z.T, x)
        inverse_weights = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        self.weights = (weights / (weights.sum() + 10 * EPS) + EPS)
        self.means = weighted_x_sum * inverse_weights


    def score_samples(self, x):

        log_support = self._log_support(x)

        lpr = log_support + np.log(self.weights)
        logprob = np.logaddexp.reduce(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])

        return logprob, responsibilities

    def predict(self, x):

        return np.sum(np.exp(self._log_support(x)), 1)

def _kmeans_init(x, k, means=None, verbose=False):

    if means is None:
        kmeans = KMeans(n_clusters=k,
                        verbose=int(verbose)).fit(x).cluster_centers_

    else:
        assert means.shape[0] >= k, 'not enough means provided for kmeans init'
        # keeping the first self.k means
        kmeans = means[:(k - 1), :]

    return kmeans

def _data_classes_mean_init(x, labels):

    n, d = x.shape

    assert labels.shape[0] == n, 'labels and data shapes must match'

    label_set = set(labels)
    n_labels = len(label_set)

    means = np.ndarray(shape=(n_labels, d))

    for l in label_set:
        matches = np.in1d(labels, l)
        means[l] = x[matches].mean(0)

    return means
