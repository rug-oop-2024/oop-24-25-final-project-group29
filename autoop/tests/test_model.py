import unittest
import numpy as np

from autoop.core.ml.model import get_model
from autoop.core.ml.model.regression.ridge_regression_model import (
    RidgeRegressionModel
)
from autoop.core.ml.model.regression.multiple_linear_regression_model import (
    MultipleLinearRegression
)
from autoop.core.ml.model.regression.lasso_regression_model import (
    LassoRegressionModel
)
from autoop.core.ml.model.classification.svm_classification_model import (
    SVMClassificationModel
)
from autoop.core.ml.model.classification.logistic_classification_model import (
    LogisticClassificationModel
)
from autoop.core.ml.model.classification.knn_classification_model import (
    KNNClassificationModel
)


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.x_reg = np.array([[1, 2], [2, 3], [3, 4]])
        self.y_reg = np.array([2, 3, 5])
        self.x_clas = np.array([[1, 2], [2, 3], [3, 4]])
        self.y_clas = np.array([2, 3, 5])

    def test_get_model(self):
        """
        Test if get model function returns the correct model
        """
        model = get_model("Ridge Regression Model")
        self.assertIsInstance(model, RidgeRegressionModel)

    def test_lasso(self):
        """
        Test if lasso works
        """
        reg = LassoRegressionModel(alpha=0.5)
        reg.fit(self.x_reg, self.y_reg)
        prediction = reg.predict(self.x_reg)
        self.assertEqual(prediction.shape, self.y_reg.shape)

    def test_ridge(self):
        """
        Test if ridge works
        """
        reg = RidgeRegressionModel(alpha=0.5)
        reg.fit(self.x_reg, self.y_reg)
        prediction = reg.predict(self.x_reg)
        self.assertEqual(prediction.shape, self.y_reg.shape)

    def test_mlr(self):
        """
        Test if multiple linear regression works
        """
        reg = MultipleLinearRegression()
        reg.fit(self.x_reg, self.y_reg)
        prediction = reg.predict(self.x_reg)
        self.assertEqual(prediction.shape, self.y_reg.shape)

    def test_knn(self):
        """
        Test if knn works
        """
        reg = KNNClassificationModel(n_neighbors=3)
        reg.fit(self.x_clas, self.y_clas)
        prediction = reg.predict(self.x_clas)
        self.assertEqual(prediction.shape, self.y_clas.shape)

    def test_svm(self):
        """
        Test if svm works
        """
        reg = SVMClassificationModel()
        reg.fit(self.x_clas, self.y_clas)
        prediction = reg.predict(self.x_clas)
        self.assertEqual(prediction.shape, self.y_clas.shape)

    def test_logistic(self):
        """
        Test if logistic works
        """
        reg = LogisticClassificationModel()
        reg.fit(self.x_clas, self.y_clas)
        prediction = reg.predict(self.x_clas)
        self.assertEqual(prediction.shape, self.y_clas.shape)
