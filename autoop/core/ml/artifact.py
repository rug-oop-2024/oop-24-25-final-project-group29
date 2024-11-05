from pydantic import BaseModel, Field
import base64
from typing import Optional, Dict, Any


class Artifact(BaseModel):
    """
    Artifact base class. Used to store and manage artifacts.

    Attributes:
    name: str
        Name of the artifact.
    type: str
        Type of artifact.
    asset_path: str
        The file path or reference location for the artifact.
    data: Any
        The data of the artifact, such as pickled objects
        or encoded files.
    metadata: Optional[dict]
        Extra metadata about the artifact's contents.
    """
    data: bytes = Field(..., description="Artifact Data")
    name: str = Field(..., description="Artifact Name")
    asset_path: str = Field(..., description="Artifact Path")
    version: str = Field(..., description="Artifact Version")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Artifact Metadata"
        )
    type: Optional[str] = Field(None, description="Artifact Type")

    def encode(self) -> str:
        """
        Encodes the artifact's binary data

        Returns:
        str: Base64 encoded string of the artifact data
        """
        if self.data is None:
            raise ValueError("No data to encode")
        return base64.b64encode(self.data).decode("utf-8")

    def decode(self, encoded_data: str) -> None:
        """
        Decodes the base64 encoded data string into binary data.

        parameters:
        encoded_data: str
            Base 64 encoded string of artifact data
        """
        try:
            self.data = base64.b64decode(encoded_data.encode("utf-8"))
        except Exception as e:
            raise ValueError(f"Failed to decode artifact data: {e}")

    def save(self, data: bytes) -> None:
        """
        Saves the artifact binary data to a given path.

        parameters:
        data: bytes
            The binary data to save
        """
        if self.data in None:
            raise ValueError("No data to save")
        if self.asset_path is None:
            raise ValueError("No path to save artifact data to")

        with open(self.asset_path, 'wb') as file:
            file.write(data)

    def read(self) -> bytes:
        """
        Load the artifact data from a given path.

        returns:
        bytes
            The binary data of the artifact
        """
        try:
            with open(self.asset_path, 'rb') as file:
                return file.read()
        except Exception as e:
            raise IOError(f"Failed to load artifact data: {e}")

    def __str__(self) -> str:
        """
        Returns a string representation of the artifact
        """
        return f"""Artifact(
        data={self.data},
        name={self.name},
        asset_path={self.asset_path},
        version={self.version},
        type={self.type}
        metadata={self.metadata}
        )"""
