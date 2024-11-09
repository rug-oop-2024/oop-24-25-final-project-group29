import streamlit as st
import pandas as pd
import io
from typing import List

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from autoop.functional.feature import detect_feature_types


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def convert_artifacts_to_datasets(artifacts: List[Artifact]) -> List[Dataset]:
    datasets = []
    for artifact in artifacts:
        datasets.append(Dataset.from_dataframe(
            name=artifact.name,
            asset_path=artifact.asset_path,
            data=pd.read_csv(io.StringIO(artifact.data.decode()))
        ))
    return datasets


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train "
    "a model on a dataset."
    )

automl = AutoMLSystem.get_instance()

datasets = convert_artifacts_to_datasets(automl.registry.list(type="dataset"))

dataset_display_names = [artifact.name for artifact in datasets]

name = st.selectbox(
    'Select a dataset',
    dataset_display_names
    )
if name:
    chosen_dataset = datasets[dataset_display_names.index(name)]

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
