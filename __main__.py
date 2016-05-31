import argparse

import numpy as np
from sklearn.cluster import KMeans

from mnist import load_mnist
import bmm
import classifier

parser = argparse.ArgumentParser(
    prog='em',
    description='train model with em'
)

parser.add_argument('--path', default='/home/data/ml/mnist',
                    help='path to the mnist data')

parser.add_argument('--k', default=10,
                    help='number of components')

args = parser.parse_args()

def compare_precisions_by_nb_of_components():

    train_data, train_labels = load_mnist(dataset='training', path=args.path)
    train_data = np.reshape(train_data, (train_data.shape[0], 784))
    train_data = np.where(train_data > 0.5, 1, 0)
    test_data, test_labels = load_mnist(dataset='testing', path=args.path)
    test_data = np.reshape(test_data, (test_data.shape[0], 784))
    test_data = np.where(test_data > 0.5, 1, 0)

    label_set = set(train_labels)

    precisions = []

    ks = list(range(1, 11)) + [15, 20, 30]

    # print('computing {} means for initialization'.format(max(ks)))
    # means = KMeans(n_clusters=max(ks),
                   # verbose=2).fit(train_data).cluster_centers_

    for k in ks:

        print('learning {} components'.format(k))

        model = classifier.classifier(k)
        model.fit(train_data, train_labels)

        predicted_labels = model.predict(test_data, label_set)
        expected_labels = test_labels

        precision = np.mean(predicted_labels == expected_labels)
        precisions.append((k, precision))
        print('precision: {}'.format(precision))

    print(precisions)

def main():

    compare_precisions_by_nb_of_components()

if __name__ == '__main__':

    main()
