import numpy as np

import bmm
import gmm

def _model_class_from_type(model_type):

    if model_type == 'bmm':
        return bmm.bmm
    elif model_type == 'gmm':
        return gmm.gmm

    raise ValueError('Unknown model type: {}'.format(model_type))

class classifier:

    def __init__(self, n_components,
                 means_init_heuristic='random',
                 model_type='bmm', means=None, verbose=False):

        self.n_components = n_components

        self.means_init_heuristic = means_init_heuristic

        self.models = dict()

        self.model_class = _model_class_from_type(model_type)

        self.means = means

        self.verbose = verbose

    def fit(self, x, labels):

        label_set = set(labels)

        for label in label_set:

            x_subset = x[np.in1d(labels, label)]

            self.models[label] = self.model_class(self.n_components,
                                                  verbose=self.verbose)

            print('training label {} ({} samples)'
                  .format(label, x_subset.shape[0]))

            self.models[label].fit(
                x_subset, means_init_heuristic=self.means_init_heuristic,
                means=self.means)

    def predict(self, x, label_set):

        highest_likelihood = 0
        likeliest_label = None

        n = x.shape[0]

        likelihoods = np.ndarray(shape=(len(label_set), n))

        for label in label_set:
            likelihoods[label] = self.models[label].predict(x)

        predicted_labels = np.argmax(likelihoods, axis=0)

        return predicted_labels
