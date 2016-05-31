import numpy as np
import argparse

from mnist import load_mnist
import bmm

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
    data = np.reshape(data, (data.shape[0], 784))
    data = np.where(data > 0.5, 1, 0)

    classifier = bmm.bmm_classifier(20, data, labels)
    classifier.train()

    test_data, test_labels = load_mnist(dataset='testing', path=args.path)
    test_data = np.reshape(test_data, (test_data.shape[0], 784))
    test_data = np.where(test_data > 0.5, 1, 0)

    labels = classifier.classify(test_data)

    print(np.mean(labels == test_labels))

if __name__ == '__main__':

    main()
