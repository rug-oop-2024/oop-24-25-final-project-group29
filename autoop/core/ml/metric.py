from abc import ABC, abstractmethod
import numpy as np


METRICS = [
    "Mean Squared Error Metric",
    "Accuracy Metric",
    "Mean Absolute Error Metric",
    "R Squared Metric",
    "AUC ROC Metric",
    "Precision Metric"
]


def get_metric(
        name: str) -> float:
    """
    Factory function to get a metric by name.

    parameters:
    name: str
        name of the metric

    Returns:
    Metric: metric instance
        instance of its given string name
    """
    if name == "Mean Squared Error Metric":
        return MeanSquaredError()
    elif name == "Accuracy Metric":
        return Accuracy()
    elif name == "Mean Absolute Error Metric":
        return MeanAbsoluteError()
    elif name == "AUC ROC Metric":
        return AucRoc()
    elif name == "R Squared Metric":
        return Rsquared()
    elif name == "Precision Metric":
        return Precision()
    else:
        raise ValueError(
            f"No metric called: {name}, Can only do metrics: {METRICS}"
            )


class Metric(ABC):
    """
    Base class for all metrics.
    remember: metrics take ground truth and ground_truth as input and
    return a real number

    Methods:
    __call__(observation: Any, ground_truth: Any) -> float:
        Calculates the metric based on the ground truth and ground_truth.

    name() -> str:
        Returns the name of the metric.
    """
    @abstractmethod
    def __call__(
        self, observation: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        Calculates the metric given the ground truth and ground_truth.

        Args:
        observation: Any
            The ground truths (x) of the model.
        ground_truth: Any
            The ground_truths (y) of the model.

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

    def evaluate(self, ground_truth, y) -> float:
        """
        Gets the metric and uses the functions __call__ and name from the
        specific metric to calculate the metric value.
        """
        return self.__call__(y, ground_truth)


class MeanSquaredError(Metric):
    """
    Metric 1 for regression.
    Class to claculate the mean squared error.
    """
    def __call__(
            self, observation: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        uses numpy to calculate the mean squared error
        """
        return np.mean((observation - ground_truth) ** 2)

    def name(self) -> str:
        return "Mean Squared Error Metric"


class Accuracy(Metric):
    """
    Metric 1 for classification.
    Class to calculate the accuracy.
    """
    def __call__(
            self, observation: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        Finds the accuracy by comparing the ground truth and ground_truth.
        """
        return np.mean(observation == ground_truth)

    def name(self) -> str:
        return "Accuracy Metric"


class MeanAbsoluteError(Metric):
    """
    Metric 2 for regression.
    Class to calculate the mean absolute error.
    """
    def __call__(
            self, observation: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        Finds the mean absolute error by getting the mean of the absolute value
        of the difference between ground truth and ground_truth.
        """
        return np.mean(np.abs(observation - ground_truth))

    def name(self) -> str:
        return "Mean Absolute Error Metric"


class AucRoc(Metric):
    """
    Metric 2 for classification.
    Class to calculate the auc_roc
    """
    def __call__(
            self, observation: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        Finds the auc_roc value by using the trapezoid rule.
        """
        sorted_indices = np.argsort(ground_truth)[::-1]
        sorted_observation = observation[sorted_indices]

        # Compute cumulative true positives and false positives
        true_positives = np.cumsum(sorted_observation)
        false_positives = np.cumsum(1 - sorted_observation)

        # Calculate TPR and FPR for each threshold
        total_positives = true_positives[-1]
        total_negatives = false_positives[-1]
        if total_positives > 0:
            true_positive_rate = true_positives / total_positives
        else:
            np.zeros_like(true_positives)

        if total_negatives > 0:
            false_positive_rate = false_positives / total_negatives
        else:
            np.zeros_like(false_positives)

        # Calculate AUC using trapezoidal integration on the TPR vs FPR
        auc = np.trapz(true_positive_rate, x=false_positive_rate)
        return auc

    def name(self) -> str:
        return "AUC ROC Metric"


class Rsquared(Metric):
    """
    Metric 3 for regression.
    Class to calculate the R Squared Metric.
    """
    def __call__(
            self, observation: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        Finds the R Squared Metric value by using the formula
        1 - residual_sum_of_squared / total_sum_of_squared
        """
        total_sum_of_squared = np.sum(
            (observation - np.mean(observation)) ** 2
            )
        residual_sum_of_squared = np.sum((observation - ground_truth) ** 2)
        return 1 - residual_sum_of_squared / total_sum_of_squared

    def name(self) -> str:
        return "R Squared Metric"


class Precision(Metric):
    """
    Metric 3 for classification.
    Class to calculate the precision.
    """
    def __call__(
            self, observation: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        Macro average precision in order to be used for more than 2 classes.
        This method calculates the average of each class and then averages.
        """
        classes = np.unique(observation)
        precision = []

        for class_ in classes:
            true_pos = np.sum(
                (observation == class_) & (ground_truth == class_)
                )
            pred_pos = np.sum(ground_truth == class_)

            if pred_pos == 0:
                precision.append(0.0)
            else:
                precision.append(true_pos / pred_pos)

        return np.mean(precision)

    def name(self) -> str:
        return "Precision Metric"
