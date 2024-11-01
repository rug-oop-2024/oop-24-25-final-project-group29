
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    """
    Class to represent the features in the dataset.

    parameters:
        name: str
            Name of the feature.
        type: Literal['numerical', 'categorical']
            Type of the feature either numerical or categorical.
    """
    name: str = Field(..., description="Feature Name")
    type: Literal["numerical", "categorical"] = Field(
        ..., description="Feature Type"
        )

    @classmethod
    def from_dataframe(
            cls, dataset: Dataset, column: str
                ) -> "Feature":
        """
        Creates a feature from a dataset instance.

        parameters:
        dataset: Dataset
            The dataset instance
        name: str
            The name of the feature

        returns:
        Feature
            The feature instance from that certain column
        """
        dataframe = dataset.read()
        if column not in dataframe.columns:
            raise ValueError(f"Column {column} not found in the dataset")

        column_type = "numerical" if np.issubdtype(
            dataframe[column].dtype, np.number
        ) else "categorical"
        return cls(name=column, type=column_type)

    def __str__(self) -> str:
        """
        Returns a string representation of the data's features (name and type)
        """
        return f"Feature(name={self.name}, type={self.type})"
