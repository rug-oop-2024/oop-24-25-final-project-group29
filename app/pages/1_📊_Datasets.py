import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")


def app():
    st.title("Datasets")
    st.write("Here the user can manage the datasets.")

    uploaded_file = st.file_uploader('Choose a csv file', type='csv')

    if st.button("Add dataset"):
        name = uploaded_file.name # REMOVE .csv
        st.write(name)
        df = pd.read_csv(uploaded_file)
        st.write(df)
        asset_path = f"datasets/{uploaded_file.name}"
        st.write(asset_path)

        dataset = Dataset.from_dataframe()
        automl.registry.register(dataset)


app()
