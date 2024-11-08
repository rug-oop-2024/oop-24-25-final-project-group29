import numpy as np

from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression.lasso_regression_model import LassoRegressionModel
from autoop.core.ml.model.regression.linear_regression_model import LinearRegressionModel
from autoop.core.ml.model.regression.ridge_regression_model import RidgeRegressionModel


class RegressionFacade(Model):
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
        super().__init__(model_type="regression")
        self._model_type = model_type.lower()
        self._alpha = alpha
        self._model = self._create_model()

    @property
    def model_type(self):
        """
        Returns the type of the regression model
        """
        return self._model_type
    
    @property
    def alpha(self):
        """
        Returns the regularization strength alpha if any
        """
        return self._alpha
    
    @property
    def model(self):
        """
        Returns the model instance or creates if there is none.
        """
        if self._model is None:
            self._model = self._create_model()
        return self._model

    def _create_model(self):
        """
        Creates the appropriate model based on the model type

        Returns:
        Model
            The model instance for the specific model type
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
