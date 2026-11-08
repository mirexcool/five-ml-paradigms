import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


class SymbolistClassifier(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=10,
        criterion="gini",
        class_weight=None,
        ccp_alpha=0.0,
        random_state=42
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

    def fit(self, X, y):
        # Train a single decision tree using CART
        self.model_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            random_state=self.random_state
        )

        self.model_.fit(X, y)

        # Store fitted model attributes
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        self.feature_importances_ = self.model_.feature_importances_

        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        return self.model_.predict_proba(X)