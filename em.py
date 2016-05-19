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
            print('iteration', i)
            self.expectation_step()
            self.maximization_step()

    def expectation_step(self):
        # update z
        pass

    def maximization_step(self):
        # update pi and parameters
        pass

class bmm_em(em):

    def __init__(self, k, x, iterations=1000, d=784):

        super().__init__(k, x, iterations)
        self.d = d

        self.mu = np.random.rand(self.k, self.d)

        for m in range(self.k):

            normalization_factor = 0.0
            for i in range(self.d):
                self.mu[m,i] = np.random.random() * 0.5 + 0.25
                normalization_factor += self.mu[m, i]
            for i in range(self.d):
                self.mu[m,i] /= normalization_factor

    def expectation_step(self):

        prod = np.zeros(self.k)

        for n in range(self.n):
            for m in range(self.k):
                t = self.pi[m]
                t *= np.prod(np.power(self.mu[m], self.x[n]))
                t *= np.prod(np.power((1.0 - self.mu[m]), (1.0 - self.x[n])))
                prod[m] = t

        s = sum(prod)
        for n in range(self.n):
            for m in range(self.k):
                if s > 0.0:
                    self.z[m,n] = prod[m] / s
                else:
                    self.z[m,n] = prod[m] / float(self.k)

    def maximization_step(self):

        for m in range(self.k):
            n_m = np.sum(self.z[m])
            self.pi[m] = n_m / self.n # update pi
            self.mu[m] = 0
            for i in range(self.n):
                self.mu[m] += self.z[m,i] * self.x[i].T
            self.mu[m] /= n_m
