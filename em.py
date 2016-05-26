import numpy as np

class em:

    def __init__(self, k, x, iterations):

        self.k = k
        self.x = x
        self.iterations = iterations

        self.n = self.x.shape[0]
        self.pi = np.array([1 / self.k for _ in range(self.k)])
        self.z = np.ndarray(shape=(self.k, self.n))

    def fit(self):

        for i in range(self.iterations):
            print('iteration', i, 'llk =', self.llk)
            self.expectation_step()
            self.maximization_step()

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

    def __init__(self, k, x, iterations=10, d=784):

        super().__init__(k, x, iterations)
        self.d = d

        self.mu = np.ndarray(shape=(self.k, self.d))

        # initialization of mu
        for m in range(self.k):
            for i in range(self.d):
                self.mu[m,i] = np.random.random() * 0.5 + 0.25

        print(self.mu)

    def expectation_step(self):

        pi = self.pi; mu = self.mu

        logsum = np.ndarray(shape=(self.k, self.n))

        for k in range(self.k):
            logsum[k, :] = np.log(pi[k]) \
                + np.log(np.prod(mu[k, :] ** self.x, 1)) \
                + np.log(np.prod((1 - mu[k, :]) ** (1 - self.x), 1))

        prod = np.exp(logsum)

        for k in range(self.k):
            self.z[k] = prod[k] / np.sum(prod[k])

    def maximization_step(self):

        n_ms = np.sum(self.z, 1)
        # updating mu
        mu = np.dot(np.diag(1 / n_ms), np.dot(self.z, self.x))
        # updating pi
        self.pi = n_ms / self.n

    @property
    def llk(self):


        return 0
