import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.linear_model import Ridge


class RidgeRegressionModel(Model):
    """
    A model that performs ridge regression.
    """
    def __init__(self, *args, **kwargs):
        """
        initialize regression model.

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

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ridge regression to the data using ridge regression formulas

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

    # def _save_model(self) -> bytes:
    #     """
    #     Saves the model parameters to a binary format.

    #     Returns
    #     bytes:
    #         serialized model parameters
    #     """
    #     parameters = np.array(
    #         [self._intercept] + list(self._coef)
    #         ).astype(np.float32)
    #     return parameters.tobytes()

    # def _load_model(self, parameters: bytes) -> None:
    #     """
    #     Load the model parameters from a binary format.
    #     """
    #     parameters = np.frombuffer(parameters, dtype=np.float32)
    #     self._intercept = parameters[0]
    #     self._coef = parameters[1:]
