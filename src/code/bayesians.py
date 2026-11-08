from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_is_fitted


class BayesianClassifier(ClassifierMixin, BaseEstimator):

    def __init__(self, var_smoothing=1e-9, priors=None):
        self.var_smoothing = var_smoothing
        self.priors = priors

    def fit(self, X, y):
        # Train Gaussian Naive Bayes
        self.model_ = GaussianNB(
            var_smoothing=self.var_smoothing,
            priors=self.priors
        )

        self.model_.fit(X, y)

        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_

        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)