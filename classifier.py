import numpy as np

import bmm

class classifier:

    def __init__(self, n_components, model_type='bmm'):

        self.n_components = n_components

        self.models = dict()

        self.model_class = bmm.bmm

    def fit(self, x, labels):

        label_set = set(labels)

        for label in label_set:

            x_subset = x[np.in1d(labels, label)]

            self.models[label] = self.model_class(self.n_components)

            print('training label {} ({} samples)'
                  .format(label, x_subset.shape[0]))

            self.models[label].fit(x_subset)

    def predict(self, x, label_set):

        highest_likelihood = 0
        likeliest_label = None

        n = x.shape[0]

        likelihoods = np.ndarray(shape=(len(label_set), n))

        for label in label_set:
            likelihoods[label] = self.models[label].predict(x)

        predicted_labels = np.argmax(likelihoods, axis=0)

        return predicted_labels
