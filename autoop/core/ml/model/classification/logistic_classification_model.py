import numpy as np
from autoop.core.ml.model.model import Model


class LogisticClassificationModel(Model):
    def __init__(self, learning_rate=0.01, max_iter=1000):
        """
        initialte the logistic regression model.
        Indludes the learning rate and the maximum iterations values.

        parameters:
        learning_rate: float
            The learning rate
        max_iter: int
            The maximum iterations
        """
        super().__init__(type="classification")
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self._coef = None
        self._intercept = None

    @property
    def coef(self) -> np.ndarray:
        """
        The model coefficients property decorator.

        returns:
        np.ndarray
            The coefficients of the logistic regression model.
        """
        return self._coef

    @coef.setter
    def coef(self, value: np.ndarray) -> None:
        self._coef = value

    @property
    def intercept(self) -> float:
        """
        The model intercept.

        returns:
        float
            The intercept of the logistic regression model.
        """
        return self._intercept

    @intercept.setter
    def intercept(self, value: float) -> None:
        self._intercept = value

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        fit the logistic regression model to the given data.
        The model is trained using sigmoid function.

        parameters:
        x: np.ndarray
            The features
        y: np.ndarray
            The predictions
        """
        n_samples, n_features = x.shape
        self._coef = np.zeros(n_features)
        self._intercept = 0

        for _ in range(self.max_iter):
            linear = np.dot(x, self.coef) + self._intercept
            predictions = self._sigmoid(linear)

            gradient_coef = (1 / n_samples) * np.dot(x.T, (predictions - y))
            gradient_intercept = (1 / n_samples) * np.sum(predictions - y)

            self._coef -= self.learning_rate * gradient_coef
            self._intercept -= self.learning_rate * gradient_intercept

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the output using the model.

        parameters:
        x: np.ndarray
            The ground truths
        returns:
        np.ndarray
            The predictions

        returns:
        np.ndarray
            The predictions
        """
        linear = np.dot(x, self._coef) + self._intercept
        return self._sigmoid(linear)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Function to execute the sigmoid function.

        parameters:
        x: np.ndarray
            The input

        returns:
        np.ndarray
            The sigmoid output
        """
        return 1 / (1 + np.exp(-x))

    # def _save_model(self) -> bytes:
    #     """
    #     serialize model parameters to bytes

    #     returns:
    #     bytes
    #         The serialized model parameters
    #     """
    #     coeficient_bytes = self._coef.tobytes()
    #     intercept_bytes = np.array(self._intercept, dtype=np.float32).tobytes()

    #     metadata = np.array([self._coef.shape[0]], dtype=np.int32).tobytes()
    #     return metadata + coeficient_bytes + intercept_bytes

    # def _load_model(self, parameters: bytes) -> None:
    #     """
    #     Deserialize the model parameters from bytes.

    #     Parameters
    #     ----------
    #     parameters : bytes
    #         The serialized model parameters.
    #     """
    #     metadata = np.frombuffer(parameters[:4], dtype=np.int32)
    #     amount_features = metadata[0]

    #     self._coef = np.frombuffer(
    #         parameters[4:4 + amount_features * 4], dtype=np.float32
    #     )
    #     self._intercept = np.frombuffer(
    #         parameters[4 + amount_features * 4:], dtype=np.float32
    #     )[0]
