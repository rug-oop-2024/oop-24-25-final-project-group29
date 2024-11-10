import streamlit as st
import pickle
import pandas as pd

from typing import Dict
from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Deployment", page_icon='🚀')


def _get_pipeline_config(base_artifact: Artifact) -> Dict:
    pipeline_data = pickle.loads(base_artifact.data)
    for artifact in pipeline_data:
        if artifact.name == "pipeline_config":
            config_data = pickle.loads(artifact.data)
            formatted_data = {
                "formatted_input_features": [
                    f"{feat.name} ({feat.type})" for feat in config_data.get(
                        "input_features", []
                        )
                ],
                "input_features": [feat.name for feat in config_data.get(
                    "input_features", []
                    )],
                "target_feature": config_data.get("target_feature").name,
                "split": config_data.get("split", 0)
            }
            return formatted_data


def _get_pipeline_model(base_artifact: Artifact) -> Model:
    pipeline_data = pickle.loads(base_artifact.data)
    for artifact in pipeline_data:
        if artifact.type == "model":
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
            pipeline_config = _get_pipeline_config(pipeline_artifact)
            if pipeline_config:
                st.subheader(f"Pipeline: {pipeline_artifact.name}")
                st.write(
                    f"Input Features: {', '.join(pipeline_config[
                        'input_features'
                        ])}"
                    )
                st.write(f"Target Feature: {pipeline_config[
                    'target_feature'
                    ]}")

                st.write(f"Train/Test Split: {pipeline_config['split']}")

    if delete_button:
        for current in selected_pipelines:
            automl.registry.delete(current.id)
        st.success("Pipelines(s) deleted!")

load_pipeline_name = st.selectbox(
    'Select a pipeline for performing predictions',
    pipeline_names
)
if load_pipeline_name:
    load_pipeline = pipeline_artifacts[
        pipeline_names.index(load_pipeline_name)
        ]
    model = _get_pipeline_model(load_pipeline)
    pipeline_config = _get_pipeline_config(load_pipeline)

    uploaded_file = st.file_uploader('Choose a csv file', type='csv')

    if st.button("Load Pipeline"):
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            asset_path = f"datasets/{uploaded_file.name}"
            dataset = Dataset.from_dataframe(
                name=uploaded_file.name,
                data=df,
                asset_path=asset_path
            )

            input_feature_names = pipeline_config["input_features"]
            input_data = df[input_feature_names]
            y = df[pipeline_config["target_feature"]]

            # Fit the model and make predictions
            model.fit(input_data, y)
            predictions = model.predict(input_data)

            st.write("Predictions:")
            prediction_df = pd.DataFrame(predictions, columns=[
                pipeline_config["target_feature"]
                ])
            st.table(prediction_df.head(10))

            st.download_button(
                label="Download Predictions CSV",
                data=prediction_df.to_csv(index=False),
                file_name="predictions.csv",
                mime="text/csv",
                key="download_predictions_csv"
            )
