
from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from typing import Literal, Dict
import pickle

from autoop.core.ml.artifact import Artifact


class Model(ABC):
    """
    Abstract base class for all models.

    Methods:
    fit(X: np.ndarray, y: np.ndarray) -> None:
        Trains the model on given data.

    predict(X: np.ndarray) -> np.ndarray:
        Makes predictions on given data.

    save(path: str) -> None:
        Saves the model to an artifact

    load(path: str) -> None:
        Loads the model from an artifact
    """
    def __init__(
            self,
            type: Literal["regression", "classification"]
            ) -> None:
        self._type = type
        self._parameters: Dict[str, np.ndarray] = {}

    @property
    def type(self) -> str:
        """
        Returns the model type.
        """
        return self._type

    @property
    def parameters(self) -> Dict[str, np.ndarray]:
        """
        Returns a deepcopy of the model parameters.
        """
        """
        Returns a deepcopy of the model parameters.
        """
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        Sets model parameters.

        Parameters:
        params: dict
            Dictionary of model parameters.
        """
        self._parameters = parameters

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def to_artifact(self, name: str) -> "Artifact":
        """
        Converts the model to an artifact

        parameters:
        name: str
            How the artifact should be named

        returns:
        Artifact
            An instance of Artifact made from the model
        """
        binary_model = pickle.dumps(self)
        artifact = Artifact(
            name=name,
            data=binary_model,
            type="model",
            metadata={self.type}
        )
        return artifact

    @staticmethod
    def from_artifact(artifact: Artifact) -> "Model":
        model = pickle.load(artifact.data)
        return model
