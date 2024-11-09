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
    elif name == "Accuracy":
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
    remember: metrics take ground truth and prediction as input and
    return a real number

    Methods:
    __call__(ground_truth: Any, prediction: Any) -> float:
        Calculates the metric based on the ground truth and prediction.

    name() -> str:
        Returns the name of the metric.
    """
    @abstractmethod
    def __call__(
        self, ground_truth: np.ndarray, prediction: np.ndarray
            ) -> float:
        """
        Calculates the metric given the ground truth and prediction.

        Args:
        ground_truth: Any
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

    def evaluate(self, prediction, y) -> float:
        """
        Gets the metric and uses the functions __call__ and name from the
        specific metric to calculate the metric value.
        """
        return self.__call__(y, prediction)


class MeanSquaredError(Metric):
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
        return "Mean Squared Error Metric"


class Accuracy(Metric):
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
        return "Accuracy Metric"


class MeanAbsoluteError(Metric):
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
        return "Mean Absolute Error Metric"


class AucRoc(Metric):
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
        sorted_indices = np.argsort(prediction)[::-1]
        sorted_ground_truth = ground_truth[sorted_indices]

        # Compute cumulative true positives and false positives
        true_positives = np.cumsum(sorted_ground_truth)
        false_positives = np.cumsum(1 - sorted_ground_truth)

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
            self, ground_truth: np.ndarray, prediction: np.ndarray
            ) -> float:
        """
        Finds the R Squared Metric value by using the formula
        1 - residual_sum_of_squared / total_sum_of_squared
        """
        total_sum_of_squared = np.sum(
            (ground_truth - np.mean(ground_truth)) ** 2
            )
        residual_sum_of_squared = np.sum((ground_truth - prediction) ** 2)
        return 1 - residual_sum_of_squared / total_sum_of_squared

    def name(self) -> str:
        return "R Squared Metric"


class Precision(Metric):
    """
    Metric 3 for classification.
    Class to calculate the precision.
    """
    def __call__(
            self, ground_truth: np.ndarray, prediction: np.ndarray
            ) -> float:
        """
        Macro average precision in order to be used for more than 2 classes.
        This method calculates the average of each class and then averages.
        """
        classes = np.unique(ground_truth)
        precision = []

        for class_ in classes:
            true_pos = np.sum(
                (ground_truth == class_) & (prediction == class_)
                )
            pred_pos = np.sum(prediction == class_)

            if pred_pos == 0:
                precision.append(0.0)
            else:
                precision.append(true_pos / pred_pos)

        return np.mean(precision)

    def name(self) -> str:
        return "Precision Metric"
