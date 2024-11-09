import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.linear_model import LinearRegression


class MultipleLinearRegression(Model):
    """
    A model that performs multiple linear regression.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(type="regression")
        self._model = LinearRegression(*args, **kwargs)
        self._hyperparameters = {
            param: value
            for param, value in self._model.get_params().items()
            if param not in ("coef_", "intercept_")
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data using the normal equation.

        Parameters:
        X: np.ndarray
            The input feature matrix of shape (n_samples, n_features).
        y: np.ndarray
            The target values of shape (n_samples,).
        """
        self._model.fit(x, y)
        self._parameters = {
            param: value
            for param, value in self._model.get_params().items()
            if param in ("coef_", "intercept_")
        }

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
        return self._model.predict(x)

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
