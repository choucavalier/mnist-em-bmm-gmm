import numpy as np

from mnist import load_mnist
import em

def main():

    data = load_mnist(dataset='training', path='/home/data/ml/mnist')
    data = np.reshape(data, (60000, 784))
    data = np.where(data > 0.5, 1, 0)

    k = 10

    model = em.bmm_em(k, data, iterations=10)
    model.fit()

if __name__ == '__main__':

    main()
