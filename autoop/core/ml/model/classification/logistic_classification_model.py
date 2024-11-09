import numpy as np
from autoop.core.ml.model.model import Model
from sklearn.linear_model import LogisticRegression


class LogisticClassificationModel(Model):
    def __init__(self, *args, **kwargs) -> None:
        """
        initialte the logistic regression model and the parameters.

        parameters:
        *args: tuple
            The arguments to pass to the super class
        **kwargs: dict
            The keyword arguments to pass to the super class
        """
        super().__init__(type="classification")
        self._model = LogisticRegression(*args, **kwargs)
        self._hyperparameters = {
            param: value
            for param, value in self._model.get_params().items()
            if param not in ("coef_", "intercept_")
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        fit the logistic regression model to the given data.

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
        return self._model.predict(x)

    # def _save_model(self) -> bytes:
    #     """
    #     serialize model parameters to bytes

    #     returns:
    #     bytes
    #         The serialized model parameters
    #     """
    #     coeficient_bytes = self._coef.tobytes()
    #     intercept_bytes = np.array(
    #     self._intercept, dtype=np.float32
    #     ).tobytes()

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
