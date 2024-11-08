import numpy as np
from autoop.core.ml.model.model import Model
from typing import Dict
from collections import Counter


class KNNClassificationModel(Model):
    """
    Class for detecting the k-nearest neighbors from a specific point
    and predicting the label of that point based on nearest neighboors.
    """
    def __init__(self, k=3):
        super().__init__(type="classification")
        self._k = k
        self._parameters = {}

    @property
    def k(self) -> int:
        """
        Returns the k value
        """
        return self._k

    @k.setter
    def k(self, value: int) -> None:
        """
        Sets the k value
        """
        if value < 1:
            raise ValueError("k must be greater than 0")
        self._k = value

    @property
    def parameters(self) -> Dict[str, np.ndarray]:
        """
        Returns the parameters of the model
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value: Dict[str, np.ndarray]) -> None:
        """
        Sets the parameters of the model
        """
        self._parameters = value

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        fit the KNN model to given data

        parameters:
        x: np.ndarray
            The ground truths
        y: np.ndarray
            The predictions
        """
        self._parameters['observations'] = x
        self._parameters['ground truths'] = y
        self.amount_observations = x.shape[0]
        self.amount_features = x.shape[1]

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
            self._parameters["observations"] - observation, axis=1
            )
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [
            self._parameters["ground truth"][i] for i in k_indices
            ]
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]

    # def _save_model(self) -> bytes:
    #     """
    #     Saves the model parameters to a binary type
    #     """
    #     observations = self._parameters["observations"]
    #     ground_truths = self._parameters["ground truth"]
    #     observations_bytes = observations.tobytes()
    #     ground_truths_bytes = ground_truths.tobytes()

    #     metadata = np.array(
    #         [self.amount_observations, self.amount_features], dtype=np.int32
    #         ).tobytes()
    #     return metadata + observations_bytes + ground_truths_bytes

    # def _load_model(self, parameters: bytes) -> None:
    #     """
    #     Load the model parameters from a binary type
    #     """
    #     metadata_size = 4 * 2
    #     metadata = np.frombuffer(parameters[:metadata_size], dtype=np.int32)
    #     self.amount_observations, self.amount_features = metadata

    #     observations = np.frombuffer(
    #         parameters[
    #             metadata_size:metadata_size
    #             + self.amount_observations * self.amount_features * 4
    #             ], dtype=np.float32
    #             )
    #     ground_truths = np.frombuffer(
    #         parameters[
    #             metadata_size + self.amount_observations
    #             * self.amount_features * 4:
    #             ], dtype=np.float32
    #             )
    #     self._parameters["observations"] = observations.reshape(
    #         self.amount_observations, self.amount_features
    #         )
    #     self._parameters["ground truth"] = ground_truths
