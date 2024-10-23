from sklearn.linear_model import LogisticRegression
from copy import deepcopy

import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from model import Model  # noqa : E402


class MultipleLogisticRegressor(Model):
    def __init__(self, penalty='l2', C=1.0):
        """
        Initialize the Logistic Regression model with various hyperparameters,
        as defined in the scikit-learn library.
        :param penalty: Type of regularization ('l1', 'l2', 'elasticnet', or
        'none')
        :param C: Inverse of regularization strength
        """
        self._model = LogisticRegression(penalty=penalty, C=C)

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the model based on the observations and labels (ground_truth)
        by applying the LogisticRegression method .fit
        """
        self._model.fit(observations, ground_truth)
        self._parameters = {
            "_coef": self._model.coef_,
            "_intercept": self._model.intercept_
        }  # Splitting the vector into weights and bias

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions based on the observations
        by applying the LogisticRegression method .predict
        """
        return self._model.predict(observations)

    @property
    def model(self) -> 'MultipleLogisticRegressor':
        """ Returns a copy of model to prevent leakage. """
        return deepcopy(self._model)
