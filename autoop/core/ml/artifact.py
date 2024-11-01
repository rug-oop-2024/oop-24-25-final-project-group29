from pydantic import BaseModel, Field
import base64
from typing import Optional, Dict


class Artifact(BaseModel):
    """
    Artifact base class. Used to store and manage artifacts.

    Attributes:
    name: str
        Name of the artifact.
    type: str
        Type of artifact.
    path: Optional[str]
        The file path or reference location for the artifact.
    data: Optional[bytes]
        The serialized binary data of the artifact, such as pickled objects
        or encoded files.
    metadata: Optional[dict]
        Extra metadata about the artifact's contents.
    """
    name: str = Field(..., description="Artifact Name")
    type: str = Field(..., description="Artifact Type")
    _path: Optional[str] = Field(None, description="Artifact Path")
    _data: Optional[bytes] = Field(None, description="Artifact Data")
    metadata: Optional[Dict] = Field(
        None, description="Artifact Metadata"
        )

    @property
    def path(self) -> Optional[str]:
        """
        getter for artifact path
        """
        return self._path

    @path.setter
    def path(self, value: str) -> None:
        """
        Setter for artifact path
        """
        self._path = value

    @property
    def data(self) -> Optional[bytes]:
        """
        getter for artifact data
        """
        return self._data

    @data.setter
    def data(self, value: bytes) -> None:
        """
        Setter for artifact data
        """
        if not isinstance(value, bytes):
            raise TypeError("Data must be in format: bytes")
        self._data = value

    def encoder(self) -> str:
        """
        Encodes the artifact's binary data

        Returns:
        str: Base64 encoded string of the artifact data
        """
        if self.data is not None:
            return base64.b64encode(self.data).decode("utf-8")
        raise ValueError("No data to encode")

    def decoder(self, encoded_data: str) -> None:
        """
        Decodes the base64 encoded data string into binary data.

        parameters:
        encoded_data: str
            Base 64 encoded string of artifact data
        """
        self.data = base64.b64decode(encoded_data.encode("utf-8"))

    def save(self, path: str) -> None:
        """
        Saves the artifact binary data to a given path.

        parameters:
        path: str
            The path to save the artifact data on
        """
        if self._data is not None:
            with open(path, 'wb') as file:
                file.write(self._data)
            self._path = path
        else:
            raise ValueError("No data to save")

    def load(self, path: str) -> None:
        """
        Load the artifact data from a given path.

        parameters:
        path: str
            The path to load the artifact data from
        """
        with open(path, 'rb') as file:
            self._data = file.read()
        self._path = path

    def to_dictionary(self) -> dict:
        """
        Converts the artifact data to a dictionary.

        returns:
        dict: A dictionary representation of the artifact data
        """
        return {
            "name": self.name,
            "type": self.type,
            "path": self._path,
            "data": self.encoder() if self._data is not None else None,
            "metadata": self.metadata
        }

    @classmethod
    def from_dictionary(cls, artifact_data: dict) -> "Artifact":
        """
        Makes an artifact instance from the dictionary.

        parameters:
        data: dict
            A dictionary of the artifact data
        """
        instance = cls(
            name=artifact_data["name"],
            type=artifact_data["type"],
            path=artifact_data.get("path"),
            metadata=artifact_data.get("metadata"),
        )
        if artifact_data.get("data"):
            instance.decoder(artifact_data["data"])
        return instance
