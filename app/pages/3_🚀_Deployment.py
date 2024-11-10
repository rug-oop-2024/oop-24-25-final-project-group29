import streamlit as st
import pickle

from typing import List
from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import get_model


def _convert_artifacts_to_pipelines(
        artifacts: List[Artifact]
        ) -> List[Pipeline]:
    pipelines = []
    for artifact in artifacts:
        pipeline_data = pickle.loads(artifact.data)
        input_features = pipeline_data["input_features"]
        target_feature = pipeline_data["target_feature"]
        split = pipeline_data["split"]


        pipeline = Pipeline(
            metrics=None,
            dataset=None,
            model=model,
            input_features=input_features,
            target_feature=target_feature,
            split=split,
        )
        pipelines.append(pipeline)

    return pipelines


automl = AutoMLSystem.get_instance()

pipeline_artifacts = automl.registry.list(type="pipeline")
pipelines = _convert_artifacts_to_pipelines(pipeline_artifacts)

st.title("Deployment")
st.write("Here the user can manage the pipelines.")

pipeline_names = [artifact.name for artifact in pipeline_artifacts]

pipeline_selected = st.selectbox(
    'Select a pipeline',
    pipeline_names
)

if pipeline_selected:
    pipeline = pipelines[pipeline_names.index(pipeline_selected)]
    st.write(pipeline)

