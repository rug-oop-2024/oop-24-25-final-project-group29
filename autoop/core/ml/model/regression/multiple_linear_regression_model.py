import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.linear_model import LinearRegression


class MultipleLinearRegression(Model):
    """
    A model that performs multiple linear regression.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the multiple linear regression model from sklearn
        LinearRegression and the parameters of the model.

        Parameters:
        args: tuple
            The arguments to pass to the super class
        kwargs: dict
            The keyword arguments to pass to the super class
        """
        super().__init__(type="regression")
        self._model = LinearRegression(*args, **kwargs)
        self._hyperparameters = {
            param: value
            for param, value in self._model.get_params().items()
            if param not in ("coef_", "intercept_")
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model to the data.

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
