
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []
    df = dataset.read()

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            features.append(Feature(name=column, type="numerical"))
        elif pd.api.types.is_string_dtype(df[column]):
            features.append(Feature(name=column, type="categorical"))

    return features
