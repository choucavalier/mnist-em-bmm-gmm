import numpy as np
from scipy import linalg

import mixture

EPS = np.finfo(float).eps

def _log_multivariate_normal_density_diag(x, means, covars):
    '''Compute Gaussian log-density at x for a diagonal model

    Taken from scikit-learn's implementation
    '''
    d = x.shape[1]
    lpr = -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(x, (means / covars).T)
                  + np.dot(x ** 2, (1.0 / covars).T))
    return lpr

def _log_multivariate_normal_density_full(x, means, covars, min_covar=1.e-7):
    '''Log probability for full covariance matrices

    Taken from scikit-learn's implementation
    '''
    n_samples, n_dim = x.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            try:
                cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                          lower=True)
            except linalg.LinAlgError:
                raise ValueError("'covars' must be symmetric, "
                                 "positive-definite")

        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = linalg.solve_triangular(cv_chol, (x - mu).T, lower=True).T
        log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                 n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob

class gmm(mixture.mixture):

    def __init__(self, n_components, covariance_type='diag',
                 n_iter=100, verbose=False):

        super().__init__(n_components, init_params='wmc', n_iter=n_iter,
                         covariance_type=covariance_type, verbose=verbose)

    def _log_support(self, x):

        if self.covariance_type == 'diag':

            lpr = _log_multivariate_normal_density_diag(
                x, self.means, self.covars)

        elif self.covariance_type == 'full':

            lpr = _log_multivariate_normal_density_full(
                x, self.means, self.covars)

        return lpr

    def _do_mstep(self, x, z):

        weights = z.sum(axis=0)
        weighted_x_sum = np.dot(z.T, x)
        norm = 1.0 / (weights[:, np.newaxis] + 10 * EPS)

        self.weights = (weights / (weights.sum() + 10 * EPS) + EPS)
        self.means = weighted_x_sum * norm

        if self.covariance_type == 'diag':
            self.covars = self._covar_mstep_diag(x, z, weighted_x_sum, norm)
        elif self.covariance_type == 'full':
            self.covars = self._covar_mstep_full(x, z, weighted_x_sum, norm)

    def _covar_mstep_diag(self, x, responsibilities, weighted_x_sum, norm):
        '''Perform the covariance M step for diagonal cases

        Taken from scikit-learn's implementation
        '''
        avg_x2 = np.dot(responsibilities.T, x * x) * norm
        avg_means2 = self.means ** 2
        avg_x_means = self.means * weighted_x_sum * norm
        return avg_x2 - 2 * avg_x_means + avg_means2 + self.min_covar

    def _covar_mstep_full(self, x, responsibilities, weighted_x_sum, norm):
        '''Perform the covariance M step for full cases

        Taken from scikit-learn's implementation
        '''
        # Eq. 12 from K. Murphy, "Fitting a Conditional Linear Gaussian
        # Distribution"
        n_features = x.shape[1]
        cv = np.empty((self.n_components, n_features, n_features))
        for c in range(self.n_components):
            post = responsibilities[:, c]
            mu = self.means[c]
            diff = x - mu
            with np.errstate(under='ignore'):
                # Underflow Errors in doing post * x.T are not important
                avg_cv = np.dot(post * diff.T, diff) / (post.sum() + 10 * EPS)
            cv[c] = avg_cv + self.min_covar * np.eye(n_features)
        return cv
