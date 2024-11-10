from typing import Optional, List
import os
import base64


class NotFoundError(Exception):
    """
    Class for handling not found errors if the path is not found.
    """
    def _init_(self, path: str) -> None:
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
        """
        Initializes the artifact class and encodes the asset path
        to a base64 string in order to create a unique id for the artifact.
        """
        self._name = name
        self._data = data
        self._type = type
        self._asset_path = asset_path
        self._metadata = metadata
        self._version = version
        self._tags = tags
        if self._asset_path:
            encoded_path = base64.b64encode(asset_path.encode()).decode()
            self._id = f"{encoded_path.replace("=", "_")}_{version}"

    @property
    def name(self) -> str:
        """
        Returns
            str: name of the artifact
        """
        return self._name

    @name.setter
    def name(self, new_name: str) -> None:
        """
        Parameters:
        new_name: str
            new name to change the artifact name to
        """
        if not isinstance(new_name, str):
            raise TypeError("new_name must be a string")
        self._name = new_name

    @property
    def data(self) -> bytes:
        """
        Returns
            bytes: data stored in the artifact
        """
        return self._data

    @property
    def type(self) -> str:
        """
        Returns
            str: type of the artifact
        """
        return self._type

    @property
    def asset_path(self) -> str:
        """
        Property getter for the asset path

        Returns:
            str: The asset path
        """
        return self._asset_path

    @property
    def metadata(self) -> set[str]:
        """
        Property getter for the metadata

        Returns:
            set[str]: The metadata
        """
        return self._metadata

    @property
    def version(self) -> str:
        """
        Property getter for the version

        Returns:
            str: The version
        """
        return self._version

    @property
    def tags(self) -> List[str]:
        """
        Property getter for the tags

        Returns:
            List[str]: The tags
        """
        return self._tags

    @property
    def id(self) -> str:
        """
        Property getter for the id

        Returns:
            str: The id
        """
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
