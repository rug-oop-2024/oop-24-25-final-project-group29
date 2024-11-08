from typing import Optional
import os
import pickle
import base64

from autoop.core.storage import Storage
from autoop.core.database import Database


class NotFoundError(Exception):
    def __init__(self, path: str) -> None:
        super().__init__(f"Path not found: {path}")


class Artifact():
    """
    Artifact base class. Used to store and manage artifacts.

    Attributes:
    name: str
        Name of the artifact
    data: bytes
        The binary data stored in the artifact
    type: str
        The type of artifact
    metadata: Optional[set[str]]
        The metadata of the artifact, optional
    version: str
        The version of the artifact
    """
    # storage = Storage()
    # db = Database(storage)

    def __init__(
            self,
            name: str,
            data: bytes,
            type: Optional[str] = "unknown",
            asset_path: Optional[str] = None,
            metadata: Optional[set[str]] = None,
            version: Optional[str] = "v_1"
            ) -> None:
        self._name = name
        self._data = data
        self._type = type
        self._asset_path = asset_path
        self._metadata = metadata
        self._version = version
        self._id = base64.b64encode(version).decode()

    @property
    def data(self) -> bytes:
        return self._data

    def _to_dict(self) -> dict:
        """Convert the Artifact instance to a dictionary."""
        return {
            "name": self._name,
            "data": base64.b64encode(self._data).decode(),
            "type": self._type,
            "metadata": list(self._metadata),
            "asset_path": self._asset_path,
            "version": self._version
        }

    @staticmethod
    def _from_dict(dict: dict) -> "Artifact":
        """Create an Artifact instance from a dictionary."""
        data = dict.get("data", "").encode("utf-8")
        return Artifact(
            name=dict.get("name"),
            data=data,
            asset_path=dict.get("asset_path"),
            type=dict.get("type"),
            metadata=set(dict.get("metadata", [])),
            version=dict.get("version")
        )

    def save(
            self,
            data: Optional[bytes] = None
            ) -> None:
        """
        Saves the artifact instance in a pickle file

        parameters:
        data: Optional[bytes]
            Optional, if the user wants to change the data in the
            artifact

        returns:
            None
        """
        if not os.path.exists(self._asset_path):
            raise NotFoundError(self._asset_path)
        # NOT GOING TO WORK, RELATIVE PATH INSTEAD
        self._asset_path += (
            f"\\{self._name.replace(" ", "_")}_"
            f"{self._version.replace(".", "_")}.pkl"
            )
        if data is not None:
            if not isinstance(data, bytes):
                raise TypeError("data has to be in bytes")
            self._data = data
        with open(self._asset_path, 'wb') as f:
            pickle.dump(self, f)

    # @staticmethod
    # def load(path: str) -> "Artifact":
    #     """
    #     Load an artifact from a pickle file

    #     parameters:
    #     path: str
    #         the path to the location of the pickle file

    #     returns:
    #     Artifact
    #         An Artifact instance of the loaded data
    #     """
    #     if not os.path.exists(path):
    #         raise NotFoundError(path)
    #     with open(path, "rb") as f:
    #         artifact = pickle.load(f)
    #     return artifact

    def read(self) -> bytes:
        """
        Returns the data stored in the Artifact

        returns:
        bytes
            The data stored in the artifact in bytes
        """
        return self._data
