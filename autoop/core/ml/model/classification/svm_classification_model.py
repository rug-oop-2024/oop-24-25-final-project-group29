import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.svm import SVC


class SVMClassificationModel(Model):
    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the regression model and parameters.

        parameters:
        *args: tuple
            The arguments to pass to the super class
        **kwargs: dict
            The keyword arguments to pass to the super class
        """
        super().__init__(type="classification")
        self._model = SVC(*args, **kwargs)
        self._hyperparameters = {
            param: value
            for param, value in self._model.get_params().items()
            if param not in ("coef_", "intercept_")
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model using SVM regression.

        parameters:
        x: np.ndarray
            The features
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
        Predict the output using this SVM model.

        parameters:
        x: np.ndarray
            The ground truths
        returns:
        np.ndarray
            The predictions
        """
        return self._model.predict(x)
