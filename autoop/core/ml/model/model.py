
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal, Dict


class Model(ABC):
    """
    Abstract base class for all models.

    Methods:
    fit(X: np.ndarray, y: np.ndarray) -> None:
        Trains the model on given data.

    predict(X: np.ndarray) -> np.ndarray:
        Makes predictions on given data.
    """
    def __init__(self, model_type: Literal["regression", "classification"]):
        self._model_type = model_type
        self._parameters: Dict[str, np.ndarray] = {}

    @property
    def model_type(self) -> str:
        """
        Returns the model type.
        """
        return self._model_type

    @property
    def parameters(self) -> Dict[str, np.ndarray]:
        return deepcopy(self._parameters)

    @parameters.setter
    def parameters(self, parameters: Dict[str, np.ndarray]) -> None:
        """
        Sets model parameters.

        Parameters:
        params: dict
            Dictionary of model parameters.
        """
        self._parameters = deepcopy(parameters)
        

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def save(self, path: str) -> None:
        """
        Saves the model to an artifact

        parameters:
        path: str
            The path to save the model on
        """
        artifact = Artifact(
            name=self.__class__.__name__,
            type="model",
            metadata={self.model_type}
            )
        artifact.data = self._save_model()
        artifact.save(path)

    def load(self, path: str) -> None:
        """
        Load the model from an artifact

        parameters:
        path: str
            The path to load the model from
        """
        artifact = Artifact.read(path)
        self._load_model(artifact.data)

    @abstractmethod
    def _save_model(self) -> bytes:
        """
        Saves the model's parameters to a binary type
        """
        pass

    @abstractmethod
    def _load_model(self, data: bytes) -> None:
        """
        Loads the model's parameters from a binary type data
        """
        pass
