from autoop.core.ml.artifact import Artifact
from abc import ABC, abstractmethod
import pandas as pd
import io


class Dataset(Artifact):

    def __init__(self, *args, **kwargs):
        """
        Initializes the dataset class and calls the super for the artifact
        class to initialize as well.

        parameters:
        args: tuple
            The arguments to pass to the super class
        kwargs: dict
            The keyword arguments to pass to the super class
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
            ) -> "Dataset":
        """
        Static method that creates a dataset from a dataframe

        parameters:
        data: pd.DataFrame
            The dataframe to create the dataset from
        name: str
            The name of the dataset
        asset_path: str
            The asset path of the dataset
        version: str
            The version of the dataset

        returns:
        Dataset
            The dataset instance created from the dataframe
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """
        the method is used to read the dataset

        returns:
        pd.DataFrame
            The dataset in the form of dataframe
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        Overrides the data with new dataframe and
        returns the encoded data in bytes

        parameters:
        data: pd.DataFrame
            The dataframe to save

        returns:
        bytes
            The encoded data in bytes
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
