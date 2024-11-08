import numpy as np

from autoop.core.ml.model.model import Model
from autoop.core.ml.model.classification.svm_classification_model import SVMClassificationModel
from autoop.core.ml.model.classification.logistic_classification_model import LogisticClassificationModel
from autoop.core.ml.model.classification.knn_classification_model import KNNClassificationModel


class ClassificationFacade(Model):
    """
    A facade for the classification models for simplified user interface.
    """
    def __init__(self, model_type: str, **kwargs):
        """
        Initializes the classification facade with a specific model type
        (SVM, logistic, KNN).

        Parameters:
        model_type: str
            The type of the classification model
        kwargs: additional parameters specific to the model.
        """
        self._model_type = model_type.lower()
        self.model = self._create_model(**kwargs)

    @property
    def model_type(self) -> str:
        """
        Returns the model type
        """
        return self._model_type

    @model_type.setter
    def model_type(self, value: str) -> None:
        """
        Sets the model type
        """
        if value.lower() not in ["svm", "logistic", "knn"]:
            raise ValueError(f"""Unknown model type: {value},
                             can only be 'svm', 'logistic', or 'knn' for
                             classification.""")
        self._model_type = value

    def _create_model(self, kwargs):
        """
        Creates the appropriate model based on the model type.
        """
        if self.model_type == "svm":
            return SVMClassificationModel(**kwargs)
        elif self.model_type == "logistic":
            return LogisticClassificationModel(**kwargs)
        elif self.model_type == "knn":
            return KNNClassificationModel(**kwargs)
        else:
            raise ValueError(f"""Unknown model type: {self.model_type},
                             can only be 'svm', 'logistic', or 'knn' for
                             classification.""")

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the model to the data.

        Parameters:
        x: np.ndarray
            The input features
        y: np.ndarray
            The labels for the input features
        """
        self.model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the output using the model.

        Parameters:
        x: np.ndarray
            The input features

        Returns:
        np.ndarray
            The predicted labels
        """
        return self.model.predict(x)

    def save(self, path: str) -> None:
        """
        Saves the model to an artifact.

        Parameters:
        path: str
            The path to save the model on
        """
        self.model.save(path)

    def load(self, path: str) -> None:
        """
        Loads the model from an artifact.

        Parameters:
        path: str
            The path to load the model from
        """
        self.model.load(path)
