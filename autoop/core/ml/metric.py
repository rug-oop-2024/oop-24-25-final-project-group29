from abc import ABC, abstractmethod
from typing import Any
import numpy as np


METRICS = [
    "mean_squared_error",
    "accuracy",
    "mean_absolute_error",
    "r_squared",
    "auc_roc",
    "precision"
]


def get_metric(name: str):
    """
    Factory function to get a metric by name.

    parameters:
    name: str
        name of the metric

    Returns:
    Metric: metric instance
        instance of its given string name
    """
    if name == "mean_squared_error":
        return mean_squared_error()
    elif name == "accuracy":
        return accuracy()
    elif name == "mean_absolute_error":
        return mean_absolute_error()
    elif name == "auc_roc":
        return auc_roc()
    elif name == "r_squared":
        return r_squared()
    elif name == "precision":
        return precision()
    else:
        raise ValueError(
            f"No metric called: {name}, Can only do metrics: {METRICS}"
            )


class Metric(ABC):
    """
    Base class for all metrics.
    remember: metrics take ground truth and prediction as input and
    return a real number

    Methods:
    __call__(ground_truth: Any, prediction: Any) -> float:
        Calculates the metric based on the ground truth and prediction.

    name() -> str:
        Returns the name of the metric.
    """
    @abstractmethod
    def __call__(self, ground_truth: Any, prediction: Any) -> float:
        """
        Calculates the metric given the ground truth and prediction.

        Args:
        ground_truth: (
            The ground truths (x) of the model.
        prediction: Any
            The predictions (y) of the model.

        Returns:
            float: The calculated metric value.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the metric
        """
        pass


class mean_squared_error(Metric):
    """
    Metric 1 for regression.
    Class to claculate the mean squared error.
    """
    def __call__(
            self, ground_truth: np.ndarray, prediction: np.ndarray
            ) -> float:
        """
        uses numpy to calculate the mean squared error
        """
        return np.mean((ground_truth - prediction) ** 2)

    def name(self) -> str:
        return "mean_squared_error"


class accuracy(Metric):
    """
    Metric 1 for classification.
    Class to calculate the accuracy.
    """
    def __call__(
            self, ground_truth: np.ndarray, prediction: np.ndarray
            ) -> float:
        """
        Finds the accuracy by comparing the ground truth and prediction.
        """
        return np.mean(ground_truth == prediction)

    def name(self) -> str:
        return "accuracy"


class mean_absolute_error(Metric):
    """
    Metric 2 for regression.
    Class to calculate the mean absolute error.
    """
    def __call__(
            self, ground_truth: np.ndarray, prediction: np.ndarray
            ) -> float:
        """
        Finds the mean absolute error by getting the mean of the absolute value
        of the difference between ground truth and prediction.
        """
        return np.mean(np.abs(ground_truth - prediction))

    def name(self) -> str:
        return "mean_absolute_error"


class auc_roc(Metric):
    """
    Metric 2 for classification.
    Class to calculate the auc_roc
    """
    def __call__(
            self, ground_truth: np.ndarray, prediction: np.ndarray
            ) -> float:
        """
        Finds the auc_roc value by using the trapezoid rule.
        """
        sorted_prediction = np.sort(prediction)
        sorted_ground_truth = ground_truth[sorted_prediction]

        cummulative_sum_t = np.cumsum(sorted_ground_truth)
        sum_t = np.sum(sorted_ground_truth)
        true_pos_rate = cummulative_sum_t / sum_t

        cummulative_sum_f = np.cumsum(1 - sorted_ground_truth)
        sum_f = np.sum(1 - sorted_ground_truth)
        falst_pos_rate = cummulative_sum_f / sum_f

        auc = np.trapezoid(true_pos_rate, falst_pos_rate)
        return auc

    def name(self) -> str:
        return "auc_roc"


class r_squared(Metric):
    """
    Metric 3 for regression.
    Class to calculate the r_squared.
    """
    def __call__(
            self, ground_truth: np.ndarray, prediction: np.ndarray
            ) -> float:
        """
        Finds the r_squared value by using the formula
        1 - residual_sum_of_squared / total_sum_of_squared
        """
        total_sum_of_squared = np.sum(
            (ground_truth - np.mean(ground_truth)) ** 2
            )
        residual_sum_of_squared = np.sum((ground_truth - prediction) ** 2)
        return 1 - residual_sum_of_squared / total_sum_of_squared

    def name(self) -> str:
        return "r_squared"


class precision(Metric):
    """
    Metric 3 for classification.
    Class to calculate the precision.
    """
    def __call__(
            self, ground_truth: np.ndarray, prediction: np.ndarray
            ) -> float:
        """
        Finds the precision as a ratio of true positives to
        true + false positives.
        """
        true_positives = np.sum((ground_truth == 1) & (prediction == 1))
        false_positives = np.sum((ground_truth == 0) & (prediction == 1))
        true_false_positives = true_positives + false_positives

        # in order not to divide by 0
        if true_false_positives == 0:
            return 0.0
        else:
            return true_positives / true_false_positives

    def name(self) -> str:
        return "precision"
