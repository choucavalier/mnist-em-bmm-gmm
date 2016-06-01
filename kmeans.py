import argparse
import pickle

import numpy as np
from sklearn.cluster import KMeans

from mnist import load_mnist

def generate_kmeans(x, k, verbose=False):

    kmeans = KMeans(n_clusters=k, verbose=int(verbose)).fit(x).cluster_centers_

    return kmeans

def load_kmeans(path):

    with open(path, 'rb+') as file:
        kmeans = pickle.load(file)

    return kmeans

def main():

    parser = argparse.ArgumentParser(
        prog='kmeans',
        description='find kmeans in data'
    )

    parser.add_argument('--path', default='/home/data/ml/mnist',
                        help='path to the mnist data')

    parser.add_argument('--k', default=10, type=int,
                        help='number of components')

    parser.add_argument('--verbose', action='store_true', default=False)

    parser.add_argument('--output', '-o', default='kmeans.dat')

    args = parser.parse_args()

    data = load_mnist(dataset='training', path=args.path, return_labels=False)
    data = np.reshape(data, (60000, 784))

    kmeans = generate_kmeans(data, int(args.k), args.verbose)

    with open(args.output, 'wb+') as file:
        pickle.dump(kmeans, file)

    print('saved kmeans in {}'.format(args.output))

if __name__ == '__main__':
    main()
