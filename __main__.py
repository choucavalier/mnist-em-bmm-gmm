import numpy as np
import argparse

from mnist import load_mnist
import em

parser = argparse.ArgumentParser(
    prog='em',
    description='train model with em'
)

parser.add_argument('--path', default='/home/data/ml/mnist',
                    help='path to the mnist data')

args = parser.parse_args()

def main():

    data = load_mnist(dataset='training', path=args.path)
    data = np.reshape(data, (60000, 784))
    data = np.where(data > 0.5, 1, 0)

    k = 10

    model = em.bmm_em(k, data, iterations=10)
    model.fit()

if __name__ == '__main__':

    main()
