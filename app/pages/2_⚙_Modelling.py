import streamlit as st
import pandas as pd
import io
from typing import List

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.metric import Metric, get_metric


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


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train "
    "a model on a dataset."
    )

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

    input_features = st.multiselect(
        "Select input features",
        options=feature_names,
    )
    target_feature = st.selectbox(
        "Select target feature",
        options=list(set(feature_names) - set(input_features))
        )

    if not input_features:
        st.warning("Please select at least one input feature.")
    if set(feature_names) == set(input_features):
        st.warning("Please leave at least one feature to serve as the target.")

    if target_feature:
        target_index = feature_names.index(target_feature)
        if features[target_index].type == "numerical":
            task_type = st.selectbox(
                "Select regression model",
                options=[
                    "Ridge Regression Model",
                    "Lasso Regression Model",
                    "Multiple Linear Regression Model"
                    ]
            )
            metric_names = st.multiselect(
                "Select a regression metric to evaluate the model",
                options=[
                    "Mean Squared Error Metric",
                    "Mean Absolute Error Metric",
                    "R Squared Metric"
                ]
            )
        else:
            task_type = st.selectbox(
                "Select classification models",
                options=[
                    "Support Vector Machine Model",
                    "Logistic Regression Model",
                    "K-Nearest Neighbors Model"
                    ]
            )
            metric_names = st.multiselect(
                "Select a classification s to evaluate the model",
                options=[
                    "Accuracy Metric",
                    "AUC ROC Metric",
                    "Precision Metric"
                ]
            )

        split = st.slider(
            "Select the percent of data for training.",
            min_value=60,
            max_value=90
        )

        if st.button("Start Pipeline"):
            pipeline = Pipeline(
                metrics=_get_metrics_list(metric_names),
                dataset=chosen_dataset,
                model=task_type,
                input_features=input_features,
                target_feature=features[target_index],
                split=split
            )
            st.write("Pipeline started successfully:")
            st.write(pipeline)