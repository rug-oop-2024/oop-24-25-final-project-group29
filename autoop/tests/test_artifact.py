import unittest

from autoop.core.ml.artifact import Artifact


class TestArtifact(unittest.TestCase):

    def setUp(self) -> None:
        artifact = Artifact(
            name="test", type="test", asset_path="test", version="test"
            )
        
