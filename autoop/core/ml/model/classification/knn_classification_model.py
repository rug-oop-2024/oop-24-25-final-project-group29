import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.neighbors import KNeighborsClassifier


class KNNClassificationModel(Model):
    """
    Class for detecting the k-nearest neighbors from a specific point
    and predicting the label of that point based on nearest neighboors.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the KNN model from sklearn
        KNeighborsClassifier and the parameters of the model.

        Parameters:
        args: tuple
            The arguments to pass to the super class
        kwargs: dict
            The keyword arguments to pass to the super class
        """
        super().__init__(type="classification")
        self._model = KNeighborsClassifier(*args, **kwargs)
        self._hyperparameters = {
            param: value
            for param, value in self._model.get_params().items()
            if param not in ("coef_", "intercept_")
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        fit the KNN model to given data.

        parameters:
        x: np.ndarray
            The ground truths
        y: np.ndarray
            The predictions
        """
        self._model.fit(x, y)
        self._parameters = {
            param: value
            for param, value in self._model.get_params().items()
            if param in ("coef_", "intercept_")
        }

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
        return self._model.predict(observation)

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
