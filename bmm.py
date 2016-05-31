import datetime

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

class bmm:

    def __init__(self, k, x, d):

        self.k = k
        self.d = d
        self.x = x
        self.xc = 1 - x

        self.n = self.x.shape[0]
        self.pi = np.array([1 / self.k for _ in range(self.k)])
        self.z = np.ndarray(shape=(self.k, self.n))
        self.mu = np.ndarray(shape=(self.k, self.d))

        # initialization of mu
        for k in range(self.k):
            for i in range(self.d):
                self.mu[k,i] = np.random.random() * 0.5 + 0.25

    def data_mean_init(self):

        mean = self.x.mean(0)

        for k in range(self.k):
            for i in range(self.d):
                self.mu[k, i] = mean[i] + np.random.random()

    def data_classes_mean_init(self, data_labels):

        labels = set(data_labels)

        assert self.k == len(labels), 'k must match the number of labels'

        for l in labels:
            matches = np.in1d(data_labels, l)
            mean = self.x[matches].mean(0)
            self.mu[l] = mean

    def fit(self):

        start = datetime.datetime.now()

        iterations = 0

        previous_log_likelihood = -np.inf
        log_likelihood = previous_log_likelihood + 1

        while iterations == 0 or (
                abs(previous_log_likelihood - log_likelihood) > 1e-5 \
                and previous_log_likelihood < log_likelihood):

            print('\riteration {} (elapsed {})'
                  .format(iterations, datetime.datetime.now() - start), end='')

            log_support = self._log_support()
            previous_log_likelihood = log_likelihood
            log_likelihood = self.log_likelihood(log_support)

            self.expectation_step(log_support)
            self.maximization_step()

            iterations += 1

        end = datetime.datetime.now()

        elapsed = end - start

        print('\r > converged in {} iterations in {}'
              .format(iterations, elapsed))


    def plot_mu(self):

        rows = self.k // 5 + 1
        columns = min(self.k, 5)

        for k in range(self.k):
            plt.subplot(rows, columns, k + 1)
            plt.imshow(scipy.misc.toimage(self.mu[k].reshape(28, 28),
                                          cmin=0.0, cmax=1.0))

    def data_mean_init(self):

        mean = self.x.mean(0)

        for k in range(self.k):
            for i in range(self.d):
                self.mu[k, i] = mean[i] * np.random.random() + 0.25
            self.mu[k] /= sum(self.mu[k])

    def _log_support(self, x=None):

        pi = self.pi; mu = self.mu

        if x is None:
            x = self.x
            xc = self.xc
        else:
            xc = 1 - x

        log_support = np.ndarray(shape=(self.k, x.shape[0]))

        for k in range(self.k):
            log_support[k, :] = np.log(pi[k]) \
                + np.sum(x * np.log(mu[k, :].clip(min=1e-50)), 1) \
                + np.sum(xc * np.log((1 - mu[k, :]).clip(min=1e-50)), 1)

        return log_support

    def expectation_step(self, log_support):

        log_normalisation = np.logaddexp.reduce(log_support, axis=0)
        log_responsibilities = log_support - log_normalisation

        self.z = np.exp(log_responsibilities)

    def maximization_step(self):

        n_ms = np.sum(self.z, 1)
        # updating mu
        self.mu = np.dot(np.diag(1 / n_ms), np.dot(self.z, self.x))
        # updating pi
        self.pi = n_ms / self.n

    def log_likelihood(self, log_support):

        return np.sum(np.log(np.sum(np.exp(log_support), 1)))

    def likelihood(self, x):

        log_support = self._log_support(x)
        return np.sum(np.exp(log_support), 0)

class bmm_classifier:

    def __init__(self, k, data, labels):

        self.data = data
        self.labels = labels
        self.label_set = set(labels)
        self.k = k

        self.n = data.shape[0]
        self.d = data.shape[1]

        self.models = dict()

    def train(self):

        for label in self.label_set:

            data_subset = self.data[np.in1d(self.labels, label)]
            self.models[label] = bmm(self.k, data_subset, self.d)

            print('training label {} ({} samples)'
                  .format(label, data_subset.shape[0]))

            self.models[label].fit()

    def classify(self, data):

        highest_likelihood = 0
        likeliest_label = None

        likelihoods = np.ndarray(shape=(len(self.label_set), data.shape[0]))
        labels = np.ndarray(shape=(data.shape[0],))

        for label in self.label_set:
            likelihoods[label] = self.models[label].likelihood(data)

        labels = np.argmax(likelihoods, axis=0)

        return labels
