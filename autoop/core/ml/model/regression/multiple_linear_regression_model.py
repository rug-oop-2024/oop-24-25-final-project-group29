import numpy as np
from autoop.core.ml.model.model import Model


class MultipleLinearRegression(Model):
    """
    A model that performs multiple linear regression.
    """
    def __init__(self):
        super().__init__(model_type="regression")
        self._coef = None
        self._intercept = None

    @property
    def coef(self) -> np.ndarray:
        """
        The model coefficients property decorator.

        Returns:
        np.ndarray
            The coefficients of the multiple linear regression model.
        """
        return self._coef

    @property
    def intercept(self) -> float:
        """
        The model intercept.

        Returns:
        float
            The intercept of the multiple linear regression model.
        """
        return self._intercept

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data using the normal equation.

        Parameters:
        X: np.ndarray
            The input feature matrix of shape (n_samples, n_features).
        y: np.ndarray
            The target values of shape (n_samples,).
        """
        x_with_ones = np.c_[np.ones((x.shape[0], 1)), x]
        self._coef = np.linalg.inv(
            x_with_ones.T @ x_with_ones
            ) @ x_with_ones.T @ y
        self._intercept = self._coef[0]
        self._coef = self._coef[1:]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output using the model.

        Parameters:
        X: np.ndarray
            The input feature matrix of shape (n_samples, n_features).

        Returns:
        np.ndarray
            The predicted target values of shape (n_samples,).
        """
        if self._coef is None:
            raise RuntimeError("Model has not been fit")

        # Add a column of ones to X for the intercept term
        x_with_ones = np.c_[np.ones((x.shape[0], 1)), x]
        # Return predictions by applying the model coefficients
        return x_with_ones @ np.r_[self._intercept, self._coef]

    # def _save_model(self) -> bytes:
    #     """
    #     Saves the model parameters to a binary type
    #     """
    #     parameters = np.array(
    #         [self._intercept] + list(self._coef)
    #         ).astype(np.float32)
    #     return parameters.tobytes()

    # def _load_model(self, parameters: bytes) -> None:
    #     """
    #     Load the model parameters from a binary type
    #     """
    #     parameters = np.frombuffer(parameters, dtype=np.float32)
    #     self._intercept = parameters[0]
    #     self._coef = parameters[1:]
