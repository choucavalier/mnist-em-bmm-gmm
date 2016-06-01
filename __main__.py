import argparse

import numpy as np
from sklearn.cluster import KMeans
import sklearn.decomposition

from mnist import load_mnist
import gmm
import classifier
import kmeans as kmeans_

parser = argparse.ArgumentParser(
    prog='em',
    description='train model with em'
)

parser.add_argument('--path', default='/home/data/ml/mnist',
                    help='path to the mnist data')

parser.add_argument('--k', default=10, type=int,
                    help='number of components')

args = parser.parse_args()

def compare_precisions_by_nb_of_components():

    kmeans = kmeans_.load_kmeans('kmeans-20.dat')

    train_data, train_labels = load_mnist(dataset='training', path=args.path)
    train_data = np.reshape(train_data, (train_data.shape[0], 784))
    test_data, test_labels = load_mnist(dataset='testing', path=args.path)
    test_data = np.reshape(test_data, (test_data.shape[0], 784))

    d = 40
    reducer = sklearn.decomposition.PCA(n_components=d)
    reducer.fit(train_data)

    train_data_reduced = reducer.transform(train_data)
    test_data_reduced = reducer.transform(test_data)
    kmeans_reduced = reducer.transform(kmeans)

    label_set = set(train_labels)

    precisions = []

    ks = list(range(1, 11)) + [15, 20, 30]

    for k in ks:

        print('learning {} components'.format(k))

        model = classifier.classifier(k, covariance_type='full',
                                      model_type='gmm',
                                      means_init_heuristic='kmeans',
                                      means=kmeans_reduced,
                                      verbose=False)
        model.fit(train_data_reduced, train_labels)

        predicted_labels = model.predict(test_data_reduced, label_set)
        expected_labels = test_labels

        precision = np.mean(predicted_labels == expected_labels)
        precisions.append((k, precision))
        print('precision: {}'.format(precision))

    print(precisions)

def main():

    compare_precisions_by_nb_of_components()

if __name__ == '__main__':

    main()
