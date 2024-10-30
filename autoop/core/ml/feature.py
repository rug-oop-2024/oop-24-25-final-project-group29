
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    """
    Class to represent the features in the dataset.

    Arguments:
        name: str
            Name of the feature.
        type: Literal['numerical', 'categorical']
            Type of the feature either numerical or categorical.
    """
    name: str = Field(..., description="Feature Name")
    type: Literal["numerical", "categorical"] = Field(
        ..., description="Feature Type"
        )

    def __str__(self) -> str:
        """
        Returns a string representation of the data's features (name and type)
        """
        return f"Feature(name={self.name}, type={self.type})"
