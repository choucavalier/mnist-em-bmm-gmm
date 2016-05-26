import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

class em:

    def __init__(self, k, d, x):

        self.k = k
        self.d = d
        self.x = x

        self.n = self.x.shape[0]
        self.pi = np.array([1 / self.k for _ in range(self.k)])
        self.z = np.ndarray(shape=(self.k, self.n))
        self.mu = np.ndarray(shape=(self.k, self.d))

    def fit(self):

        for iteration in range(10):
            print('iteration', iteration, 'llk =', self.llk)
            for k in range(self.k):
                im = self.plot_mu(k)
                im.save('/tmp/iteration{}-{}.png'.format(iteration, k))
            self.expectation_step()
            self.maximization_step()

    def plot_mu(self, k):

        return scipy.misc.toimage(self.mu[k].reshape(28, 28),
                                  cmin=0.0, cmax=1.0)

    def expectation_step(self):
        # update z
        pass

    def maximization_step(self):
        # update pi and parameters
        pass

    @property
    def llk(self):
        # compute log likelihood
        pass

class bmm_em(em):

    def __init__(self, k, x, d):

        super().__init__(k, d, x)

        # initialization of mu
        for m in range(self.k):
            for i in range(self.d):
                self.mu[m,i] = np.random.random() * 0.5 + 0.25

    def expectation_step(self):

        pi = self.pi; mu = self.mu

        logsum = np.ndarray(shape=(self.k, self.n))

        for k in range(self.k):
            logsum[k, :] = np.log(pi[k]) \
                + np.sum(np.log(mu[k, :] ** self.x), 1) \
                + np.sum(np.log((1 - mu[k, :]) ** (1 - self.x)), 1)

        prod = np.exp(logsum)

        for k in range(self.k):
            self.z[k] = prod[k] / np.sum(prod[k])

    def maximization_step(self):

        n_ms = np.sum(self.z, 1)
        # updating mu
        self.mu = np.dot(np.diag(1 / n_ms), np.dot(self.z, self.x))
        # updating pi
        self.pi = n_ms / self.n

        print(self.mu)

    @property
    def llk(self):


        return 0
