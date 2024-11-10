import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.linear_model import Ridge


class RidgeRegressionModel(Model):
    """
    A model that performs ridge regression.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        initialize regression model from sklearn Ridge and
        the parameters of the model.

        parameters:
        alpha: float
            The regularization strenght
        """
        super().__init__(type="regression")
        self._model = Ridge(*args, **kwargs)
        self._hyperparameters = {
            param: value
            for param, value in self._model.get_params().items()
            if param not in ("coef_", "intercept_")
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ridge regression to the data using sklearn ridge regression

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
        return self._model.predict(x)
