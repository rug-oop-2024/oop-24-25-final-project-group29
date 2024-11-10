import unittest
from unittest.mock import patch, mock_open
import base64
from autoop.core.ml.artifact import Artifact, NotFoundError


class TestArtifact(unittest.TestCase):

    def setUp(self) -> None:
        """
        Setup for the tests. Creates an instance of Artifact to be used
        in multiple tests.
        """
        self.data = b"test binary data"
        self.artifact = Artifact(
            name="test_artifact",
            data=self.data,
            type="model",
            asset_path="test/path/to/asset",
            metadata={"metadata1", "metadata2"},
            version="v_1",
            tags=["tag1", "tag2"]
        )

    def test_initialization(self) -> None:
        """
        Test the initialization of the Artifact object.
        """
        self.assertEqual(self.artifact.name, "test_artifact")
        self.assertEqual(self.artifact.data, self.data)
        self.assertEqual(self.artifact.type, "model")
        self.assertEqual(self.artifact.asset_path, "test/path/to/asset")
        self.assertEqual(self.artifact.metadata, {"metadata1", "metadata2"})
        self.assertEqual(self.artifact.version, "v_1")
        self.assertEqual(self.artifact.tags, ["tag1", "tag2"])
        encoded_path = base64.b64encode("test/path/to/asset".encode()).decode()
        test_id = f"{encoded_path.replace("=", "_")}_v_1"
        self.assertEqual(self.artifact.id, test_id)

    def test_name_setter(self) -> None:
        """
        Test the setter for the name property.
        """
        self.artifact.name = "new_name"
        self.assertEqual(self.artifact.name, "new_name")

        with self.assertRaises(TypeError):
            self.artifact.name = 123

    def test_save_method_invalid_path(self):
        """
        Test that a NotFoundError is raised when the asset path does not exist.
        """
        with patch("os.path.exists", return_value=False):
            with self.assertRaises(NotFoundError):
                self.artifact.save(self.data)

    def test_read_method(self):
        """
        Test the read method for reading the data from the artifact.
        """
        self.assertEqual(self.artifact.read(), self.data)


if __name__ == "__main__":
    unittest.main()
