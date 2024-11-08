import numpy as np
from autoop.core.ml.model.model import Model


class SVMClassificationModel(Model):
    def __init__(self, C: float = 1.0) -> None:
        """
        Initializes the regression model.

        parameters:
        C: float
            The regularization strength
        """
        super().__init__(type="classification")
        self.C = C
        self._coef = None
        self._intercept = None

    @property
    def coef(self) -> np.ndarray:
        """
        The model coefficients property decorator.

        returns:
        np.ndarray
            The coefficients of the linear regression model.
        """
        return self._coef

    @coef.setter
    def coef(self, value: np.ndarray) -> None:
        """
        Sets the model coefficients property decorator.

        parameters:
        value: np.ndarray
            The coefficients of the regression model.
        """
        self._coef = value

    @property
    def intercept(self) -> float:
        """
        The model intercept.

        returns:
        float
            The intercept of the linear regression model.
        """
        return self._intercept

    @intercept.setter
    def intercept(self, value: float) -> None:
        """
        Sets the model intercept property decorator.

        parameters:
        value: float
            The intercept of the regression model.
        """
        self._intercept = value

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
        self._coef = np.zeros(amount_features)
        self._intercept = 0

        for _ in range(1000):
            for i in range(amount_samples):
                condition = y[i] * np.dot(
                    x[i], self._coef
                    ) + self._intercept >= 1
                if condition:
                    self._coef -= self.C * (self._coef / amount_samples)
                else:
                    self._coef += self.C * (y[i] * x[i] / amount_samples)

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
        decision = np.dot(x, self._coef) + self._intercept
        return np.where(decision >= 0, 1, -1)

    # def _save_model(self) -> bytes:
    #     """
    #     Save the model coefficient and intercept to bytes
    #     """
    #     coef_in_bytes = self._coef.tobytes()
    #     intercept_in_bytes = np.array(
    #         self._intercept, dtype=np.float32
    #         ).tobytes()

    #     return coef_in_bytes + intercept_in_bytes

    # def _load_model(self, data: bytes) -> None:
    #     """
    #     Loads the model's parameters from a binary type data
    #     """
    #     amount_features = len(data) // (4 + 4)
    #     self._coef = np.frombuffer(
    #         data[:amount_features * 4], dtype=np.float32
    #         )
    #     self._intercept = np.frombuffer(
    #         data[amount_features * 4:], dtype=np.float32
    #         )[0]
