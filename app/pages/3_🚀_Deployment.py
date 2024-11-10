import streamlit as st

from typing import List
from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.pipeline import Pipeline


def _convert_artifacts_to_pipelines(artifacts: List[Artifact]) -> List[Pipeline]:




automl = AutoMLSystem.get_instance()

pipeline_artifacts = automl.registry.list(type="pipeline")
pipelines = [artifact.name for artifact in pipeline_artifacts]

st.title("Deployment")
st.write("Here the user can manage the pipelines.")

pipeline_selected = st.selectbox(
    'Select a pipeline',
    pipelines
)

"""
Show the pipeline summary including name, version,
metrics, input features and target feature, model, and split.
"""
if pipeline_selected:
    pipeline = automl.registry.get(pipeline_selected)
    st.write(pipeline)
