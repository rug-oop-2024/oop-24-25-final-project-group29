import numpy as np

from autoop.core.ml.model.model import (
    Model
)
from autoop.core.ml.model.regression.lasso_regression_model import (
    LassoRegressionModel
)
from autoop.core.ml.model.regression.multiple_linear_regression_model import (
    MultipleLinearRegression
)
from autoop.core.ml.model.regression.ridge_regression_model import (
    RidgeRegressionModel
)


class RegressionFacade(Model):
    """
    A facade for simplified user interface to choose regression or
    classification models.
    """
    def __init__(self, model_name: str, **kwargs) -> None:
        """
        Initializes the model facade with a specific model type and name.

        parameters:
        type: str
            The type of the model ('regression' or 'classification')
        model_name: str
            The name of the model (e.g., 'linear', 'lasso', 'ridge')
        **kwargs:
            The additional parameters specific to the model.
        """
        super().__init__(type="regression")
        self._model_name = model_name
        self._model = self._create_model()

    @property
    def model_name(self) -> str:
        """
        Returns the name of the specific model.
        """
        return self._model_name

    @property
    def alpha(self) -> float:
        """
        Returns the regularization strength alpha if applicable.
        """
        return self._alpha

    @property
    def model(self) -> Model:
        """
        Returns the model instance or creates it if none exists.
        """
        if self._model is None:
            self._model = self._create_model()
        return self._model

    def _create_model(self) -> Model:
        """
        Creates the appropriate model based on the model name and type.

        Returns:
        Model
            The model instance for the specific model type.
        """
        if self.type == "regression":
            if self.model_name == "Multiple Linear Regression Model":
                return MultipleLinearRegression()
            elif self.model_name == "Lasso Regression Model":
                return LassoRegressionModel(alpha=self.alpha)
            elif self.model_name == "Ridge Regression Model":
                return RidgeRegressionModel(alpha=self.alpha)
            else:
                raise ValueError(
                    f"Unknown model name: {self.model_name}. "
                    f"Available options: 'Multiple Linear Regression Model',"
                    f" 'Lasso Regression Model', 'Ridge Regression Model'"
                )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the data.

        parameters:
        x: np.ndarray
            The input data
        y: np.ndarray
            The target data
        """
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the output using the model.

        parameters:
        x: np.ndarray
            The input data

        returns:
        np.ndarray
            The predictions
        """
        return self.model.predict(x)

    def save(self, path: str) -> None:
        """
        Saves the model to an artifact.

        parameters:
        path: str
            The path to save the model on
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """
        Loads the model from an artifact.

        parameters:
        path: str
            The path to load the model from
        """
        self.model.load(path)
