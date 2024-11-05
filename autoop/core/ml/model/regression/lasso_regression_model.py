import numpy as np
from autoop.core.ml.model.model import Model


class LassoRegressionModel(Model):
    def __init__(self, alpha=1.0):
        """
        initialize regression model.

        parameters:
        alpha: float
            The regularization strenght
        """
        super().__init__(model_type="regression")
        self.alpha = alpha
        self.coef = None
        self.intercept = None

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
        self.coef = np.zeros(amount_feature)
        self.intercept = 0

        # gradient descent values needed
        learning_rate = 0.01
        max_iter = 1000

        for _ in range(max_iter):
            predictions = self.predict(x)
            residual = predictions - y

            for i in range(amount_feature):
                gradient = x[:, i].T @ residual / amount_sample
                if self.coef[i] > 0:
                    self.coef[i] = max(
                        self.coef[i] - learning_rate * (
                            gradient + self.alpha
                            ), 0
                        )
                else:
                    self.coef[i] = min(
                        self.coef[i] - learning_rate * (
                            gradient - self.alpha
                            ), 0
                        )
            self.intercept = np.mean(y - x @ self.coef)

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
        if self.coef is None:
            raise RuntimeError("Model has not been fit")

        return x @ self.coef + self.intercept

    def _save_model(self) -> bytes:
        """
        Saves the model parameters to a binary type
        """
        parameters = np.array(
            [self.intercept] + list(self.coef)
            ).astype(np.float32)
        return parameters.tobytes()

    def _load_model(self, parameters: bytes) -> None:
        """
        Load the model parameters from a binary type
        """
        parameters = np.frombuffer(parameters, dtype=np.float32)
        self.intercept = parameters[0]
        self.coef = parameters[1:]
