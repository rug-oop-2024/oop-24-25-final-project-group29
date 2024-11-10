import unittest
import numpy as np
from autoop.core.ml.metric import (
    get_metric,
    MeanSquaredError,
    MacroRecall,
    Accuracy,
    MeanAbsoluteError,
    Rsquared,
    Precision
)


class TestMetric(unittest.TestCase):

    def setUp(self) -> None:
        # Regression data
        # (used for mean_squared_error, mean_absolute_error, and r_squared)
        self.ground_truth_reg = np.array([1.5, 2.0, 3.0, 4.0])
        self.prediction_reg = np.array([1.4, 2.1, 3.2, 3.9])

        # Classification data
        # (used for accuracy, auc_roc, and precision)
        self.ground_truth_clas = np.array([1, 0, 1, 1, 0, 1])
        self.prediction_clas = np.array([1, 0, 0, 1, 0, 1])

    def test_get_metric(self) -> None:
        """
        Test if the get_metric function returns the correct metric
        """
        metric_name = "Mean Squared Error Metric"
        metric = get_metric(metric_name)
        self.assertIsInstance(metric, MeanSquaredError)

        metric_name = "Macro Recall Metric"
        metric = get_metric(metric_name)
        self.assertIsInstance(metric, MacroRecall)

    def test_mean_squared_error(self) -> None:
        """
        Test if the mean_squared_error function returns the correct value.
        """
        mse = MeanSquaredError()
        result = mse(self.ground_truth_reg, self.prediction_reg)
        expected = np.mean((self.ground_truth_reg - self.prediction_reg) ** 2)
        self.assertAlmostEqual(result, expected, places=5)

    def test_recall(self) -> None:
        """
        Test if the macro recall function returns the correct value.
        """
        recall = MacroRecall()
        ground_truth = np.array([0, 0, 1, 1])
        prediction = np.array([0, 1, 1, 0])
        expected = 0.5
        result = recall(prediction, ground_truth)
        self.assertAlmostEqual(result, expected, places=2)

    def test_accuracy(self) -> None:
        """
        Test if the accuracy function returns the correct value.
        """
        acc = Accuracy()
        result = acc(self.ground_truth_clas, self.prediction_clas)
        expected = np.mean(self.ground_truth_clas == self.prediction_clas)
        self.assertAlmostEqual(result, expected, places=5)

    def test_mean_absolute_error(self) -> None:
        """
        Test if the mean_absolute_error function returns the correct value.
        """
        mae = MeanAbsoluteError()
        result = mae(self.ground_truth_reg, self.prediction_reg)
        expected = np.mean(np.abs(self.ground_truth_reg - self.prediction_reg))
        self.assertAlmostEqual(result, expected, places=5)

    def test_rsquared(self) -> None:
        """
        Test if the rsquared function returns the correct value.
        """
        rsq = Rsquared()
        result = rsq(self.ground_truth_reg, self.prediction_reg)
        total_sum_of_squares = np.sum(
            (self.ground_truth_reg - np.mean(self.ground_truth_reg)) ** 2
            )
        residual_sum_of_squares = np.sum(
            (self.ground_truth_reg - self.prediction_reg) ** 2
            )
        expected = 1 - (residual_sum_of_squares / total_sum_of_squares)
        self.assertAlmostEqual(result, expected, places=5)

    def test_precision(self) -> None:
        """
        Test if the precision function returns the correct value.
        """
        prec = Precision()
        result = prec(self.ground_truth_clas, self.prediction_clas)

        classes = np.unique(self.ground_truth_clas)
        precisions = [
            np.sum(
                (self.ground_truth_clas == c) & (self.prediction_clas == c)
                ) /
            np.sum(
                self.prediction_clas == c
                ) if np.sum(self.prediction_clas == c) > 0 else 0
            for c in classes
        ]
        expected = np.mean(precisions)
        self.assertAlmostEqual(result, expected, places=5)
