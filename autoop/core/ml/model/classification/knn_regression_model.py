import numpy as np
from autoop.core.ml.model.model import Model
from typing import Dict
from collections import Counter


class KNNRegressionModel(Model):
    """
    Class for detecting the k-nearest neighbors from a specific point
    and predicting the label of that point based on nearest neighboors.
    """
    @property
    def parameters(self) -> Dict[str, np.ndarray]:
        return self.parameters

    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.parameters = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        fit the KNN model to given data

        parameters:
        x: np.ndarray
            The ground truths
        y: np.ndarray
            The predictions
        """
        self.parameters['observations'] = x
        self.parameters['ground truths'] = y

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """
        predict the output using the model.

        parameters:
        observation: np.ndarray
            The input features

        returns:
        np.ndarray
            The predicted label for the input obserbations
        """
        predictions = np.array([self._predict_single(x) for x in observation])
        return predictions

    def _predict_single(self, observation: np.ndarray) -> int:
        """
        Predict the label of a single observation.

        parameters:
        observation: np.ndarray
            The input features

        returns:
        most_common: int
            The most common label from the neighboors for the input obserbation
        """
        # I GOT FROM ASSIGNMENT 1
        distances = np.linalg.norm(
            self.parameters["Observations"] - observation, axis=1
            )
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [
            self.parameters["Ground truth"][i] for i in k_indices
            ]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
