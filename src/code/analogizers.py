from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted


class AnalogizerClassifier(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        n_neighbors=11,
        weights="distance",
        metric="minkowski",
        p=2
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        self.model_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p
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