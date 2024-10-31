from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import (
    linear_regression_model as LinearRegressionModel,
    ridge_regression_model as RidgeRegressionModel,
    lasso_regression_model as LassoRegressionModel,
    MultipleLinearRegression
)
from autoop.core.ml.model.classification import (
    logistic_regression_model as LogisticRegressionModel,
    knn_regression_model as KNNModel,
    DecisionTreeModel,
)


REGRESSION_MODELS = [
    "linear_regression",
    "ridge_regression",
    "lasso_regression",
    "multiple_linear_regression"
]

CLASSIFICATION_MODELS = [
    "logistic_regression",
    "knn_regression",
    "decision_tree"
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
        elif model_name == "linear_regression":
            return LinearRegressionModel()
        elif model_name == "ridge_regression":
            return RidgeRegressionModel()
        elif model_name == "lasso_regression":
            return LassoRegressionModel()

    elif model_name in CLASSIFICATION_MODELS:
        if model_name == "logistic_regression":
            return LogisticRegressionModel()
        elif model_name == "knn_regression":
            return KNNModel()
        # better not to do decision tree it looks bery hard
        elif model_name == "decision_tree":
            return DecisionTreeModel()
    else:
        raise ValueError(
            f"No model called: {model_name}, Can only do models: {
                REGRESSION_MODELS + CLASSIFICATION_MODELS
                }"
            )
