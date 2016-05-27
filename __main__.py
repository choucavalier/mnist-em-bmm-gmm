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

parser.add_argument('--k', default=10,
                    help='number of components')

args = parser.parse_args()

def main():

    data, labels = load_mnist(dataset='training', path=args.path)
    data = np.reshape(data, (60000, 784))
    data = np.where(data > 0.5, 1, 0)

    model = em.bmm(int(args.k), data, 784)
    # model.data_mean_init()
    model.fit()

if __name__ == '__main__':

    main()
