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
        super().__init__(type="regression")
        self._alpha = alpha
        self._coef = None
        self._intercept = None

    @property
    def alpha(self) -> float:
        """
        Returns the regularization strength.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        """
        Sets the regularization strength.
        """
        if value < 0:
            raise ValueError("Alpha must be greater than 0")
        self._alpha = value

    @property
    def coef(self) -> np.ndarray:
        """
        Returns the coefficients of the model.
        """
        return self._coef

    @property
    def intercept(self) -> float:
        """
        Returns the intercept of the model.
        """
        return self._intercept

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ridge regression to the data using ridge regression formulas

        parameters:
        x: np.ndarray
            The ground truths
        y: np.ndarray
            The predictions
        """
        x_with_ones = np.c_[np.ones((x.shape[0], 1)), x]
        amount_of_features = x_with_ones.shape[1]
        identity_matrix = np.identity(amount_of_features)
        identity_matrix[0, 0] = 0

        self._coef = np.linalg.inv(
            x_with_ones.T @ x_with_ones + self.alpha * identity_matrix
            ) @ x_with_ones.T @ y
        self._intercept = self._coef[0]
        self._coef = self._coef[1:]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output using the model.

        Parameters:
        x: np.ndarray
            The input features

        Returns:
        np.ndarray
            The predictions
        """
        if self._coef is None:
            raise RuntimeError("Model has not been fit")

        return self._intercept + x @ self._coef

    def _save_model(self) -> bytes:
        """
        Saves the model parameters to a binary format.

        Returns
        bytes:
            serialized model parameters
        """
        parameters = np.array(
            [self._intercept] + list(self._coef)
            ).astype(np.float32)
        return parameters.tobytes()

    def _load_model(self, parameters: bytes) -> None:
        """
        Load the model parameters from a binary format.
        """
        parameters = np.frombuffer(parameters, dtype=np.float32)
        self._intercept = parameters[0]
        self._coef = parameters[1:]
