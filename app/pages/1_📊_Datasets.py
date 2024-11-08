import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


def app():
    st.title("Datasets")
    st.write("This page is for managing datasets. Here you can add, remove, or use existing datasets.")

    # Add a file uploader
    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

    if st.button('Add Dataset'):
        name = uploaded_file.name
        st.write(name)
        path = f"datasets/{uploaded_file.name}"
        st.write(path)
        df = pd.read_csv(uploaded_file)
        st.write(df)
        dataset = Dataset.from_dataframe(name=name, asset_path=path, data=df)
        automl.registry.register(dataset)

    # st.subheader("List of Datasets")
    # datasets = automl.registry.list(type="dataset")
    # for dataset in datasets:
    #     st.write(f"{dataset.name}")

    #     if st.button(f"Delete {dataset.name}", key=dataset.name):
    #         automl.registry.delete(dataset)
    #         st.experimental_rerun()


app()
