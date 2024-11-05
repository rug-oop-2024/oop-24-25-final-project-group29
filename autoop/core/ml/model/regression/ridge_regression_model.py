import numpy as np
from autoop.core.ml.model.model import Model


class RidgeRegressionModel(Model):
    """
    A model that performs ridge regression.
    """
    def __init__(self, alpha=1.0):
        """
        initialize regression model.

        parameters:
        alpha: float
            The regularization strenght
        """
        super().__init__(model_type="regression")
        self.alpha = alpha
        self.coef = None
        self.intercept = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ridge regression to the data using ridge regression formulas

        parameters:
        x: np.ndarray
            The ground truths
        y: np.ndarray
            The predictions
        """
        x_with_ones = np.c_[np.ones(x.shape[0], 1), x]
        amount_of_features = x_with_ones.shape[1]
        identity_matrix = np.identity(amount_of_features)
        identity_matrix[0, 0] = 0

        self.coef = np.linalg.inv(
            x_with_ones.T @ x_with_ones + self.alpha * identity_matrix
            ) @ x_with_ones.T @ y
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
        Saves the model parameters to a binary format.
        """
        parameters = np.array(
            [self.intercept] + list(self.coef)
            ).astype(np.float32)
        return parameters.tobytes()

    def _load_model(self, parameters: bytes) -> None:
        """
        Load the model parameters from a binary format.
        """
        parameters = np.frombuffer(parameters, dtype=np.float32)
        self.intercept = parameters[0]
        self.coef = parameters[1:]
