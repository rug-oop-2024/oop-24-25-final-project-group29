
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Assumption: only categorical and numerical features and no NaN values.
    This function is used to detect is the feature types are numerical
    or categorical.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []
    df = dataset.read()

    for column in df.columns:
        feature_type = _detect_feature_type(df[column])
        features.append(Feature(name=column, type=feature_type))
    return features


def _detect_feature_type(column):
    """
    This helper method is used to determine the feature type of one column
    at a time.
    """
    if pd.api.types.is_numeric_dtype(column):
        if column.nunique() > 5:
            return "categorical"
        return "numerical"
    elif pd.api.types.is_string_dtype(column):
        return "categorical"
    else:
        raise TypeError(f"Unsupported feature type: {type(column)}")
