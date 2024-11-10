import streamlit as st
import pandas as pd
import io
from typing import List

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.model import Model
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import (
    Metric, get_metric, CLASSIFICATION_METRICS, REGRESSION_METRICS
    )
from autoop.core.ml.model import (
    get_model, CLASSIFICATION_MODELS, REGRESSION_MODELS
    )


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def _convert_artifacts_to_datasets(artifacts: List[Artifact]) -> List[Dataset]:
    datasets = []
    for artifact in artifacts:
        datasets.append(Dataset.from_dataframe(
            name=artifact.name,
            asset_path=artifact.asset_path,
            data=pd.read_csv(io.StringIO(artifact.data.decode()))
        ))
    return datasets


def _get_metrics_list(metric_names: List[str]) -> List[Metric]:
    metric_list = []
    for metric_name in metric_names:
        metric_list.append(get_metric(metric_name))
    return metric_list


def _get_pipeline_artifact(pipeline: Pipeline) -> Artifact:
    for artifact in pipeline.artifacts:
        if artifact.name == "pipeline_config":
            return artifact


st.write("# ⚙ Modelling")

automl = AutoMLSystem.get_instance()

datasets = _convert_artifacts_to_datasets(automl.registry.list(type="dataset"))

dataset_display_names = [artifact.name for artifact in datasets]

dataset_name = st.selectbox(
    'Select a dataset',
    dataset_display_names
    )

if dataset_name:
    chosen_dataset = datasets[dataset_display_names.index(dataset_name)]

    features = detect_feature_types(chosen_dataset)
    feature_names = [feature.name for feature in features]

    input_features_names = st.multiselect(
        "Select input features",
        options=feature_names,
    )
    target_feature_name = st.selectbox(
        "Select target feature",
        options=list(set(feature_names) - set(input_features_names))
        )

    if not input_features_names:
        st.warning("Please select at least one input feature.")
    if set(feature_names) == set(input_features_names):
        st.warning("Please leave at least one feature to serve as the target.")

    if target_feature_name and input_features_names:
        target_index = feature_names.index(target_feature_name)
        target_feature = features[target_index]

        input_features = []
        for feature in features:
            if feature.name in input_features_names:
                input_features.append(feature)

        if features[target_index].type == "numerical":
            task_type = st.selectbox(
                "Select regression model",
                options=REGRESSION_MODELS
            )
            metric_names = st.multiselect(
                "Select a regression metric to evaluate the model",
                options=REGRESSION_METRICS
            )
        else:
            task_type = st.selectbox(
                "Select classification models",
                options=CLASSIFICATION_MODELS
            )
            metric_names = st.multiselect(
                "Select a classification to evaluate the model",
                options=CLASSIFICATION_METRICS
            )

        split = st.slider(
            "Select the percent of data used for training.",
            min_value=60,
            max_value=90
        )

        pipeline = Pipeline(
            metrics=_get_metrics_list(metric_names),
            dataset=chosen_dataset,
            model=get_model(task_type),
            input_features=input_features,
            target_feature=target_feature,
            split=split/100
        )

        pipeline_name = st.text_input(
                    "Enter a name for this pipeline"
                    )
        pipeline_version = st.text_input(
                "Enter a version for this pipeline"
                )
        if pipeline_name:
            pipeline_asset_path = f"pipelines/{pipeline_name}.pkl"
            st.write(pipeline_asset_path)

        execute_pipeline_button = False
        save_pipeline_button = False
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Execute Pipeline"):
                execute_pipeline_button = True
        with col2:
            if st.button("Save Pipeline"):
                save_pipeline_button = True

        if execute_pipeline_button:
            st.markdown(f"""
                # Summary

                📊 **Metrics**: {metric_names} \n
                🗂️ **Dataset**: {chosen_dataset.name} \n
                🤖 **Model**: {task_type} \n
                🔍 **Input Features**: {input_features_names} \n
                🎯 **Target Feature**: {target_feature_name} \n
                ✂️ **Train/Test Split**: {split}%/{100-split}%
                """, unsafe_allow_html=True)
            result = pipeline.execute()
            st.write(result)

        if save_pipeline_button:
            if not pipeline_name:
                st.warning(
                    "Please enter a name for the pipeline before you save"
                    )
            if not pipeline_version:
                st.warning(
                    "Please enter a version for the pipeline before you save"
                    )
            if pipeline_name and pipeline_version:
                st.write(pipeline_asset_path)
                pipeline_artifact = _get_pipeline_artifact(pipeline)
                user_pipeline_artifact = Artifact(
                    name=pipeline_name,
                    data=pipeline_artifact.data,
                    type="pipeline",
                    asset_path=pipeline_asset_path,
                )
                automl.registry.register(user_pipeline_artifact)
                st.success("Pipeline saved successfully")
