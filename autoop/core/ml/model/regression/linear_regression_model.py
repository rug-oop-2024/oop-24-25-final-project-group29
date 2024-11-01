import numpy as np
from autoop.core.ml.model.model import Model


class LinearRegressionModel(Model):
    """
    A model that performs linear regression.
    """
    def __init__(self):
        super().__init__(model_type="regression")
        self.coef = None
        self.intercept = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the data to the model using normal equation.

        parameters:
        x: np.ndarray
            The ground truths
        y: np.ndarray
            The predictions
        """
        X_with_ones = np.c_[np.ones(x.shape[0], 1), x]
        self.coef = np.linalg.inv(
            X_with_ones.T @ X_with_ones
            ) @ X_with_ones.T @ y
        self.intercept = self.coef[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output using the model.

        parameters:
        x: np.ndarray
            The ground truths
        returns:
        np.ndarray
            The predictions
        """
        if self.coef is None:
            raise RuntimeError("Model has not been fit")

        x_with_ones = np.c_[np.ones(x.shape[0], 1), x]
        return x_with_ones @ self.coef

    def _save_model(self) -> bytes:
        """
        Saves the model parameters to a binary type
        """
        parameters = np.array(
            [self.intercept] + list(self.coef)
            ).astype(np.float32)
        return parameters.tobytes()

    def _load_model(self, parameters: bytes) -> None:
        """
        Load the model parameters from a binary type
        """
        parameters = np.frombuffer(parameters, dtype=np.float32)
        self.intercept = parameters[0]
        self.coef = parameters[1:]
