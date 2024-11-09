from typing import Optional, List
import os
import base64


class NotFoundError(Exception):
    def _init_(self, path: str) -> None:
        super()._init_(f"Path not found: {path}")


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
    def __init__(
            self,
            name: str,
            data: bytes,
            type: Optional[str] = "unknown",
            asset_path: Optional[str] = None,
            metadata: Optional[set[str]] = None,
            version: Optional[str] = "v_1",
            tags: Optional[List[str]] = None
            ) -> None:
        self._name = name
        self._data = data
        self._type = type
        self._asset_path = asset_path
        self._metadata = metadata
        self._version = version
        self._tags = tags
        encoded_path = base64.b64encode(asset_path.encode()).decode()
        self._id = f"{encoded_path.replace("=", "_")}_{version}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def data(self) -> bytes:
        return self._data

    @property
    def type(self) -> str:
        return self._type

    @property
    def asset_path(self) -> str:
        return self._asset_path

    @property
    def metadata(self) -> set[str]:
        return self._metadata

    @property
    def version(self) -> str:
        return self._version

    @property
    def tags(self) -> List[str]:
        return self._tags

    @property
    def id(self) -> str:
        return self._id

    def save(self, data: bytes) -> None:
        """
        Saves new data to the asset path

        parameters:
        data: bytes
            The new data to be saved at the asset path

        returns:
            None
        """
        if not os.path.exists(self.asset_path):
            raise NotFoundError(self.asset_path)
        if not isinstance(data, bytes):
            raise TypeError("data has to be in bytes")
        self._data = data
        with open(self.asset_path, 'wb') as f:
            f.write(data)

    def read(self) -> bytes:
        """
        Returns the data stored in the Artifact

        returns:
        bytes
            The data stored in the artifact in bytes
        """
        return self._data
