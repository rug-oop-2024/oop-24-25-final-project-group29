from typing import List, Dict
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 ) -> None:
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical":
            if model.type != "classification":
                raise ValueError(
                    "Model type must be classification for categorical feature"
                    )
        if target_feature.type == "continuous":
            if model.type != "regression":
                raise ValueError(
                    "Model type must be regression for continuous feature"
                    )

    def __str__(self) -> str:
        """
        Method to return a string representation of the pipeline
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Property get method.

        returns:
            Model
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Used to get the artifacts generated during the pipeline execution
        to be saved.

        returns:
            List[Artifact]
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(
            Artifact(name="pipeline_config", data=pickle.dumps(pipeline_data))
            )
        artifacts.append(
            self._model.to_artifact(name=f"pipeline_model_{self._model.type}")
            )
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """
        Used to register artifacts that are made during the pipeline process.

        parameters:
        name: str
            The name of the artifact
        artifact: Artifact
            The artifact being registered
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocesses the feature data.
        """
        (
            target_feature_name, target_data, artifact
            ) = preprocess_features([self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
            )
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
            ]

    def _split_data(self) -> None:
        """
        Used to split the data into training and testing sets
        """
        # Split the data into training and testing sets
        split = self._split
        self._train_X = [
            vector[:int(split * len(vector))] for vector in self._input_vectors
            ]
        self._test_X = [
            vector[int(split * len(vector)):] for vector in self._input_vectors
            ]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))
            ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):
            ]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Helper method to concatenate vectors into a single array
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Used to train the model on the training set
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Used to evaluate the model on the test set
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> Dict:
        """
        Executes a full pipeline. By preprocessing features, splitting
        the data, training the model and evaluating it on both train
        and test sets.

        returns:
        dict
            A dictionary with train and test set predictions and metrics.
        """
        self._preprocess_features()
        self._split_data()

        if self._train_y.ndim > 1 and self._train_y.shape[1] == 1:
            self._train_y = self._train_y.ravel()
        elif self._train_y.ndim > 1:
            self._train_y = np.argmax(self._train_y, axis=1)

        self._train()
        self._evaluate()

        testset_metrics = [
            (metric.name(), result) for metric, result in self._metrics_results
        ]
        testset_predictions = self._predictions

        trainset_predictions = self._model.predict(
            self._compact_vectors(self._train_X)
        )
        trainset_metrics = [
            (metric.name(), metric.evaluate(
                trainset_predictions, self._train_y
                )) for metric in self._metrics
            ]

        return {
            "train metrics": trainset_metrics,
            "test metrics": testset_metrics,
            "train predictions": trainset_predictions,
            "test predictions": testset_predictions
        }
