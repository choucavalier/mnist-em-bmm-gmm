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

        for m in range(self.k):
            for i in range(self.d):
                self.mu[m, i] = mean[i] + np.random.random()

    def data_classes_mean_init(self, data_labels):

        labels = set(data_labels)

        assert self.k == len(labels), 'k must match the number of labels'

        for l in labels:
            matches = np.in1d(data_labels, l)
            mean = self.x[matches].mean(0)
            self.mu[l] = mean

    def fit(self):

        for iteration in range(1, 11):
            for k in range(self.k):
                im = self.plot_mu(k)
                im.save('/tmp/iteration{}-{}.png'.format(iteration, k))

            log_support = self._log_support()
            log_likelihood = self.log_likelihood(log_support)

            print('iteration {} - llk = {}'.format(iteration, log_likelihood))

            self.expectation_step(log_support)
            self.maximization_step()


    def plot_mu(self, k):

        return scipy.misc.toimage(self.mu[k].reshape(28, 28),
                                  cmin=0.0, cmax=1.0)

    def data_mean_init(self):

        mean = self.x.mean(0)

        for k in range(self.k):
            for i in range(self.d):
                self.mu[k, i] = mean[i] * np.random.random() + 0.25
            self.mu[k] /= sum(self.mu[k])

    def _log_support(self):

        pi = self.pi; mu = self.mu

        log_support = np.ndarray(shape=(self.k, self.n))

        for k in range(self.k):
            log_support[k, :] = np.log(pi[k]) \
                + np.sum(self.x * np.log(mu[k, :].clip(min=1e-50)), 1) \
                + np.sum(self.xc * np.log((1 - mu[k, :]).clip(min=1e-50)), 1)

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
