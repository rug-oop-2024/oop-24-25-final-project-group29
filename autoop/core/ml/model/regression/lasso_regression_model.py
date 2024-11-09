import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso


class LassoRegressionModel(Model):
    def __init__(self, *args, **kwargs):
        """
        initialize regression model.

        parameters:
        alpha: float
            The regularization strenght
        """
        super().__init__(type="regression")
        self._model = Lasso(*args, **kwargs)
        self._hyperparameters = {
            param: value
            for param, value in self._model.get_params().items()
            if param not in ("coef_", "intercept_")
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the data to the model using normal equation.

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

        parameters:
        x: np.ndarray
            The ground truths

        returns:
        np.ndarray
            The predictions
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
