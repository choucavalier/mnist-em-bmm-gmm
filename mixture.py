from datetime import datetime
import sys

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

EPS = np.finfo(float).eps

class mixture:

    def __init__(self, n_components, n_iter=100, tol=1e-3, verbose=False):

        #: number of components in the mixture
        self.n_components = n_components
        #: max number of iterations
        self.n_iter = n_iter
        #: convergence threshold
        self.tol = tol
        self.verbose = verbose

        k = self.n_components

        self.weights = np.array([1 / k for _ in range(k)])

        self.converged_ = False

    def fit(self, x):

        k = self.n_components
        n = x.shape[0]
        d = x.shape[1]

        self.means = np.ndarray(shape=(k, d))

        # initialization of the means
        for i in range(k):
            for j in range(d):
                self.means[i, j] = np.random.random() * 0.5 + 0.25

        start = datetime.now()

        iterations = 0

        prev_log_likelihood = None
        current_log_likelihood = -np.inf

        for i in range(1, self.n_iter + 1):

            elapsed = datetime.now() - start

            prev_log_likelihood = current_log_likelihood

            # expectation step
            log_likelihoods, responsibilities = self.score_samples(x)
            current_log_likelihood = log_likelihoods.mean()

            if self.verbose:

                print('[{:02d}] likelihood = {} (elapsed {})'
                      .format(i, current_log_likelihood, elapsed))

            if prev_log_likelihood is not None:
                change = abs(current_log_likelihood - prev_log_likelihood)
                if change < self.tol:
                    self.converged_ = True
                    break

            self._do_mstep(x, responsibilities)

            iterations += 1

        end = datetime.now()

        elapsed = end - start

        print('converged in {} iterations in {}'.format(i, elapsed))

    def plot_means(self):

        k = self.n_components

        rows = k // 5 + 1
        columns = min(k, 5)

        for i in range(k):
            plt.subplot(rows, columns, i + 1)
            plt.imshow(scipy.misc.toimage(self.means[i].reshape(28, 28),
                                          cmin=0.0, cmax=1.0))

    def score_samples(self, x):

        log_support = self._log_support(x)
        lpr = log_support + np.log(self.weights)
        logprob = np.logaddexp.reduce(lpr, axis=1)
        responsibilities = np.exp(lpr - logprob[:, np.newaxis])

        return logprob, responsibilities

    def predict(self, x):

        return np.sum(np.exp(self._log_support(x)), 1)

def _kmeans_mu_init(k, means=None):

    if means is None:
        kmeans = KMeans(n_clusters=k).fit(self.x).cluster_centers_

    else:
        # keeping the first self.k means
        kmeans = means[:(k - 1), :]

    return kmeans

def _data_classes_mean_init(x, labels):

    n, d = x.shape

    assert labels.shape[0] == n, 'labels and data shapes must match'

    label_set = set(data_labels)
    n_labels = len(label_set)

    means = np.ndarray(shape=(n_labels, d))

    for l in label_set:
        matches = np.in1d(labels, l)
        means[l] = x[matches].mean(0)

    return means
