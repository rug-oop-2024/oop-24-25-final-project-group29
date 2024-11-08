from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import (
    ridge_regression_model as RidgeRegressionModel,
    lasso_regression_model as LassoRegressionModel,
    multiple_linear_regression_model as MultipleLinearRegression
)
from autoop.core.ml.model.classification import (
    knn_classification_model as KNNModel,
    logistic_classification_model as LogisticRegressionModel,
    svm_classification_model as SvmClassificationModel,
)


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
    if model_name in REGRESSION_MODELS:
        if model_name == "multiple_linear_regression":
            return MultipleLinearRegression()
        elif model_name == "ridge_regression":
            return RidgeRegressionModel()
        elif model_name == "lasso_regression":
            return LassoRegressionModel()

    elif model_name in CLASSIFICATION_MODELS:
        if model_name == "logistic_regression":
            return LogisticRegressionModel()
        elif model_name == "knn_regression":
            return KNNModel()
        elif model_name == "svm":
            return SvmClassificationModel()
    else:
        raise ValueError(
            f"No model called: {model_name}, Can only do models: {
                REGRESSION_MODELS + CLASSIFICATION_MODELS
                }"
            )
