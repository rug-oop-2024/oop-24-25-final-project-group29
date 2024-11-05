import numpy as np

from autoop.core.ml.model.regression.lasso_regression_model import LassoRegressionModel
from autoop.core.ml.model.regression.linear_regression_model import LinearRegressionModel
from autoop.core.ml.model.regression.ridge_regression_model import RidgeRegressionModel


class regressionFacade:
    """
    A facade for the regression models for simplified user interface.
    """
    def __init__(self, model_type: str, alpha: float = 1.0):
        """
        Initializes the regression facade with a specific model type
        (linear, lasso, ridge).

        parameters:
        model_type: str
            The type of the regression model
        alpha: float
            The regularization strength if any
        """
        self.model_type = model_type.lower()
        self.alpha = alpha
        self.model = self._create_model()

    def _create_model(self):
        """
        Creates the appropriate model based on the model type
        """
        if self.model_type == "linear":
            return LinearRegressionModel()
        elif self.model_type == "lasso":
            return LassoRegressionModel(alpha=self.alpha)
        elif self.model_type == "ridge":
            return RidgeRegressionModel(alpha=self.alpha)
        else:
            raise ValueError(f"""Unknown model type: {self.model_type},
                             can only be linear, lasso, or ridge for
                             regression""")

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the data.

        parameters:
        x: np.ndarray
            The ground truths
        y: np.ndarray
            The predictions
        """
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the output using the model.

        parameters:
        x: np.ndarray
            The ground truths

        returns:
        np.ndarray
            The predictions
        """
        return self.model.predict(x)

    def save(self, path: str) -> None:
        """
        Saves the model to an artifact

        parameters:
        path: str
            The path to save the model on
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """
        Loads the model from an artifact

        parameters:
        path: str
            The path to load the model from
        """
        self.model.load(path)
