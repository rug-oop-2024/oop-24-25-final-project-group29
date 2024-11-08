import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.linear_model import Lasso

class LassoRegressionModel(Model):
    def __init__(self, alpha=1.0):
        """
        initialize regression model.

        parameters:
        alpha: float
            The regularization strenght
        """
        super().__init__(type="regression")
        self._alpha = alpha
        self.model = Lasso(alpha=self.alpha)

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

    @property
    def coef(self) -> np.ndarray:
        """
        Returns the coefficients of the model.
        """
        return self._coef

    @property
    def intercept(self) -> float:
        """
        Returns the intercept of the model.
        """
        return self._intercept

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the data to the model using normal equation.

        parameters:
        x: np.ndarray
            The ground truths

        y: np.ndarray
            The predictions
        """
        amount_sample, amount_feature = x.shape
        self._coef = np.zeros(amount_feature)
        self._intercept = 0

        # gradient descent values needed
        learning_rate = 0.01
        max_iter = 1000

        for _ in range(max_iter):
            predictions = self.predict(x)
            residual = predictions - y

            for i in range(amount_feature):
                gradient = x[:, i].T @ residual / amount_sample
                if self._coef[i] > 0:
                    self._coef[i] = max(
                        self._coef[i] - learning_rate * (
                            gradient + self._alpha
                            ), 0
                        )
                else:
                    self._coef[i] = min(
                        self._coef[i] - learning_rate * (
                            gradient - self._alpha
                            ), 0
                        )
            self._intercept = np.mean(y - x @ self._coef)

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
        if self._coef is None:
            raise RuntimeError("Model has not been fit")

        return x @ self._coef + self._intercept

    def _save_model(self) -> bytes:
        """
        Saves the model parameters to a binary type
        """
        parameters = np.array(
            [self._intercept] + list(self._coef)
            ).astype(np.float32)
        return parameters.tobytes()

    def _load_model(self, parameters: bytes) -> None:
        """
        Load the model parameters from a binary type
        """
        parameters = np.frombuffer(parameters, dtype=np.float32)
        self._intercept = parameters[0]
        self._coef = parameters[1:]
