import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class EvolutionaryClassifier(ClassifierMixin, BaseEstimator):

    # Model settings
    def __init__(
        self,
        n_features=40,
        n_iter=150,
        n_pop=60,
        class_0_weight=5.0,
        weight_bound=1.0,
        bias_bound=2.0,
        random_state=42
    ):
        self.n_features = n_features
        self.n_iter = n_iter
        self.n_pop = n_pop
        self.class_0_weight = class_0_weight
        self.weight_bound = weight_bound
        self.bias_bound = bias_bound
        self.random_state = random_state


    # Train the model using a genetic algorithm
    def fit(self, X, y):

        from pyBlindOpt.ga import genetic_algorithm

        # Convert input data to NumPy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # Check that X is a table and y has two classes
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("Only binary classification is supported")

        # Convert target classes to 0 and 1
        binary_y = (y == self.classes_[1]).astype(float)

        # Count the number of input features
        self.n_features_in_ = X.shape[1]

        # Define allowed ranges for feature weights
        bounds = []

        for i in range(self.n_features_in_):
            bounds.append([-self.weight_bound, self.weight_bound])

        # Add the allowed range for the bias
        bounds.append([-self.bias_bound, self.bias_bound])

        bounds = np.array(bounds, dtype=float)


        # Calculate how bad a candidate set of weights is
        def objective(params):

            # Separate feature weights and bias
            weights = params[:-1]
            bias = params[-1]

            # Calculate the linear model output
            z = X @ weights + bias
            z = np.clip(z, -500, 500)

            # Calculate the error for class 0
            loss_class_0 = (
                self.class_0_weight
                * (1 - binary_y)
                * np.logaddexp(0, z)
            )

            # Calculate the error for class 1
            loss_class_1 = (
                binary_y
                * np.logaddexp(0, -z)
            )

            # Return the average error
            total_loss = loss_class_0 + loss_class_1

            return float(np.mean(total_loss))


        # Find the best feature weights and bias
        best_params, best_loss = genetic_algorithm(
            objective=objective,
            bounds=bounds,
            n_iter=self.n_iter,
            n_pop=self.n_pop,
            seed=self.random_state,
            verbose=False
        )

        # Save learned parameters inside the model
        self.coef_ = np.asarray(best_params[:-1], dtype=float)
        self.intercept_ = float(best_params[-1])
        self.optimization_loss_ = float(best_loss)

        return self


    # Calculate the raw model score
    def decision_function(self, X):

        if not hasattr(self, "coef_"):
            raise ValueError("The model must be trained first")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Unexpected input shape")

        return X @ self.coef_ + self.intercept_


    # Convert scores into class probabilities
    def predict_proba(self, X):

        scores = self.decision_function(X)

        scores = np.clip(scores, -500, 500)

        probability_class_1 = 1 / (1 + np.exp(-scores))
        probability_class_0 = 1 - probability_class_1

        return np.column_stack([
            probability_class_0,
            probability_class_1
        ])


    # Predict class 0 or class 1
    def predict(self, X):

        probabilities = self.predict_proba(X)

        predicted_class_indices = (
            probabilities[:, 1] >= 0.5
        ).astype(int)

        return self.classes_[predicted_class_indices]