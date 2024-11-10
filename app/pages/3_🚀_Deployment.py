import streamlit as st
import pickle

from typing import Dict
from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact


st.set_page_config(page_title="Deployment", page_icon='🚀')


def _get_pipeline_config(base_artifact: Artifact) -> Dict:
    pipeline_artifacts = pickle.loads(base_artifact.data)
    for artifact in pipeline_artifacts:
        if artifact.name == "pipeline_config":
            return pickle.loads(artifact.data)


automl = AutoMLSystem.get_instance()

st.title("Deployment")
st.write("Here the user can manage the pipelines.")

pipeline_artifacts = automl.registry.list(type="pipeline")

pipeline_names = [artifact.name for artifact in pipeline_artifacts]

selected_pipeline_names = st.multiselect(
    'Select pipelines to manage',
    pipeline_names
    )
if selected_pipeline_names:
    selected_pipelines = []
    for name in selected_pipeline_names:
        selected_pipelines.append(
            pipeline_artifacts[pipeline_names.index(name)]
            )

    col1, col2 = st.columns(2)
    view_button = False
    delete_button = False
    with col1:
        if st.button("View Pipelines"):
            view_button = True
    with col2:
        if st.button("Delete Selected Pipelines"):
            delete_button = True

    if view_button:
        for pipeline_artifact in selected_pipelines:
            st.write(_get_pipeline_config(pipeline_artifact))
    if delete_button:
        for current in selected_pipelines:
            automl.registry.delete(current.id)
        st.success("Pipelines(s) deleted!")
        st.rerun()

load_pipeline_name = st.selectbox(
    'Select a pipeline for performing predictions',
    pipeline_names
)
