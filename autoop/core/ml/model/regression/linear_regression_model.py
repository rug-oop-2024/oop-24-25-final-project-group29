import numpy as np
from autoop.core.ml.model.model import Model


class LinearRegressionModel(Model):
    """
    A model that performs linear regression.
    """
    def __init__(self):
        super().__init__(model_type="regression")
        self._coef = None
        self._intercept = None

    @property
    def coef(self) -> np.ndarray:
        """
        The model coefficients property decorator.

        returns:
        np.ndarray
            The coefficients of the linear regression model.
        """
        return self._coef

    @property
    def intercept(self) -> float:
        """
        The model intercept.

        returns:
        float
            The intercept of the linear regression model.
        """
        return self._intercept

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
        self._coef = np.linalg.inv(
            X_with_ones.T @ X_with_ones
            ) @ X_with_ones.T @ y
        self._intercept = self._coef[0]

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
        if self._coef is None:
            raise RuntimeError("Model has not been fit")

        x_with_ones = np.c_[np.ones((x.shape[0], 1)), x]
        return x_with_ones @ self._coef

    def _save_model(self) -> bytes:
        """
        Saves the model parameters to a binary type
        """
        parameters = np.array(
            [self._intercept] + list(self._coef)
            ).astype(np.float32)
        return parameters.tobytes()

    def _load_model(self, parameters: bytes) -> None:
        """
        Load the model parameters from a binary type
        """
        parameters = np.frombuffer(parameters, dtype=np.float32)
        self._intercept = parameters[0]
        self._coef = parameters[1:]
