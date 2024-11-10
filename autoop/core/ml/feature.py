from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset


class Feature():
    """
    Class to represent the features in the dataset.

    parameters:
        name: str
            Name of the feature.
        type: Literal['numerical', 'categorical']
            Type of the feature either numerical or categorical.
    """
    def __init__(
            self,
            name: str,
            type: Literal["numerical", "categorical"]) -> None:
        self._name = name
        self._type = type

    @property
    def name(self) -> str:
        return self._name

    @property
    def type(self) -> str:
        return self._type

    @staticmethod
    def from_dataframe(dataset: Dataset, column: str) -> "Feature":
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
        df = dataset.read()
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in the dataset")

        column_type = "numerical" if np.issubdtype(
            df[column].dtype, np.number
        ) else "categorical"
        return Feature(name=column, type=column_type)

    def __str__(self) -> str:
        """
        Returns a string representation of the data's features (name and type)
        """
        return f"Feature(name={self.name}, type={self.type})"
