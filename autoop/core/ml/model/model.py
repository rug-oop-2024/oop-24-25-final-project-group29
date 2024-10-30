
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy
from typing import Literal


class Model(ABC):
    """
    Abstract base class for all models.

    Methods:
    fit(X: np.ndarray, y: np.ndarray) -> None:
        Trains the model on given data.

    predict(X: np.ndarray) -> np.ndarray:
        Makes predictions on given data.
    """
    # IDK HOW WE WANT THESE IF DICT OR DIFFERENT (ITS FROM OUR ASSINGMENT 1)
    # _parameters: Dict[str, np.ndarray] = PrivateAttr()

    # @property
    # @abstractmethod
    # def parameters(self) -> Dict[str, np.ndarray]:
    #     pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    # ONLY IMPLEMENT WHEN I DO SAVERS
    # @abstractmethod
    # def save(self, path: str) -> None:
    #     pass

    # @abstractmethod
    # def load(self, path: str) -> None:
    #     pass
