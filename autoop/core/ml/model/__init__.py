from autoop.core.ml.model.model import Model

from autoop.core.ml.model.classification.knn_classification_model import KNNClassificationModel
from autoop.core.ml.model.classification.svm_classification_model import SVMClassificationModel
from autoop.core.ml.model.classification.logistic_classification_model import LogisticClassificationModel
from autoop.core.ml.model.regression.lasso_regression_model import LassoRegressionModel
from autoop.core.ml.model.regression.multiple_linear_regression_model import MultipleLinearRegression
from autoop.core.ml.model.regression.ridge_regression_model import RidgeRegressionModel


REGRESSION_MODELS = [
    "ridge",
    "lasso",
    "mlr"
]

CLASSIFICATION_MODELS = [
    "logistic",
    "knn",
    "svm"
]


def get_model(model_name: str) -> Model:
    """
    Factory function to get a model by name.

    parameters:
    model_name: str
        name of the model

    Returns:
    Model: model instance
        instance of its given string name
    """
    model_name = model_name.lower()
    if model_name in REGRESSION_MODELS:
        if model_name == "mlr":
            return MultipleLinearRegression()
        elif model_name == "ridge":
            return RidgeRegressionModel()
        elif model_name == "lasso":
            return LassoRegressionModel()

    elif model_name in CLASSIFICATION_MODELS:
        if model_name == "logistic":
            return LogisticClassificationModel()
        elif model_name == "knn":
            return KNNClassificationModel()
        elif model_name == "svm":
            return SVMClassificationModel()
    else:
        raise ValueError(
            f"No model called: {model_name}, Can only do models: {
                REGRESSION_MODELS + CLASSIFICATION_MODELS
                }"
            )
