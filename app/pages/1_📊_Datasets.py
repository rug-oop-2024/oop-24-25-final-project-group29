import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
dataset_display_names = [artifact.name for artifact in datasets]


st.title("Datasets")
st.write("Here the user can manage the datasets.")

name = st.selectbox(
    'Select a dataset',
    dataset_display_names
    )
if name:
    dataset = datasets[dataset_display_names.index(name)]

uploaded_file = st.file_uploader('Choose a csv file', type='csv')

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    asset_path = f"datasets/{uploaded_file.name}"
    dataset = Dataset.from_dataframe(
        name=uploaded_file.name,
        data=df,
        asset_path=asset_path,
    )
    if st.button("Add dataset"):
        automl.registry.register(dataset)
        st.success("Dataset added!")
        st.rerun()
