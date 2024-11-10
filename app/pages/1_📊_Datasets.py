import streamlit as st
import pandas as pd
import io

from typing import List
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact


st.set_page_config(page_title="Datasets", page_icon='📊')


def _convert_artifacts_to_datasets(artifacts: List[Artifact]) -> List[Dataset]:
    datasets = []
    for artifact in artifacts:
        datasets.append(Dataset.from_dataframe(
            name=artifact.name,
            asset_path=artifact.asset_path,
            data=pd.read_csv(io.StringIO(artifact.data.decode()))
        ))
    return datasets


automl = AutoMLSystem.get_instance()

dataset_artifacts = automl.registry.list(type="dataset")
datasets = _convert_artifacts_to_datasets(dataset_artifacts)
dataset_display_names = [artifact.name for artifact in datasets]


st.title("Datasets")
st.write("Here the user can manage the datasets.")

uploaded_file = st.file_uploader('Choose a csv file', type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    asset_path = f"datasets/{uploaded_file.name}"
    dataset = Dataset.from_dataframe(
        name=uploaded_file.name,
        data=df,
        asset_path=asset_path
    )
    if st.button("Add dataset"):
        automl.registry.register(dataset)
        st.success("Dataset added!")
        st.rerun()

dataset_name = st.selectbox(
    'Select datasets to manage',
    dataset_display_names
    )
if dataset_name:
    selected_dataset = datasets[dataset_display_names.index(dataset_name)]

    col1, col2 = st.columns(2)
    view_button = False
    delete_button = False
    with col1:
        if st.button("View Dataset"):
            view_button = True
    with col2:
        if st.button("Delete Selected Dataset"):
            delete_button = True

    if view_button:
        st.dataframe(selected_dataset.read())
    if delete_button:
        automl.registry.delete(selected_dataset.id)
        st.success("Dataset deleted!")
