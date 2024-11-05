import numpy as np
from autoop.core.ml.model.model import Model


class SVMRegressionModel(Model):
    def __init__(self, C: float = 1.0) -> None:
        """
        Initializes the regression model.

        parameters:
        C: float
            The regularization strength
        """
        super().__init__(model_type="classification")
        self.C = C
        self.coef = None
        self.intercept = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model using SVM regression.

        parameters:
        x: np.ndarray
            The features
        y: np.ndarray
            The predictions
        """
        amount_samples, amount_features = x.shape
        self.coef = np.zeros(amount_features)
        self.intercept = 0

        for _ in range(1000):
            for i in range(amount_samples):
                condition = y[i] * np.dot(
                    x[i], self.coef
                    ) + self.intercept >= 1
                if condition:
                    self.coef -= self.C * (self.coef / amount_samples)
                else:
                    self.coef += self.C * (y[i] * x[i] / amount_samples)

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
        decision = np.dot(x, self.coef) + self.intercept
        return np.where(decision >= 0, 1, -1)

    def _save_model(self) -> bytes:
        """
        Save the model coefficient and intercept to bytes
        """
        coef_in_bytes = self.coef.tobytes()
        intercept_in_bytes = np.array(
            self.intercept, dtype=np.float32
            ).tobytes()

        return coef_in_bytes + intercept_in_bytes

    def _load_model(self, data: bytes) -> None:
        """
        Loads the model's parameters from a binary type data
        """
        amount_features = len(data) // (4 + 4)
        self.coef = np.frombuffer(data[:amount_features * 4], dtype=np.float32)
        self.intercept = np.frombuffer(
            data[amount_features * 4:], dtype=np.float32
            )[0]
