from abc import ABC, abstractmethod
import numpy as np

REGRESSION_METRICS = [
    "Mean Squared Error Metric",
    "Mean Absolute Error Metric",
    "R Squared Metric"
]

CLASSIFICATION_METRICS = [
    "Accuracy Metric",
    "Macro Recall Metric",
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
    elif name == "Macro Recall Metric":
        return MacroRecall()
    elif name == "R Squared Metric":
        return Rsquared()
    elif name == "Precision Metric":
        return Precision()
    else:
        raise ValueError(
            f"No metric called: {name}, can only do metrics: "
            f"{REGRESSION_METRICS}{CLASSIFICATION_METRICS}"
            )


class Metric(ABC):
    """
    Base class for all metrics.
    remember: metrics take observations and ground_truth as input and
    return a real number

    Abstract methods:
    __call__(observations: Any, ground_truth: Any) -> float:
        Calculates the metric based on the observations and ground_truth.

    name() -> str:
        Returns the name of the metric.
    """
    @abstractmethod
    def __call__(
        self, observations: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        Calculates the metric given the ground truth and ground_truth.

        Args:
        observations: Any
            The observations (x) of the model.
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

    def evaluate(
            self, predictions: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        Gets the metric and uses the functions __call__ and name from the
        specific metric to calculate the metric value.
        """
        if ground_truth.ndim > 1 and ground_truth.shape[1] == 1:
            ground_truth = ground_truth.ravel()
        elif ground_truth.ndim > 1:
            ground_truth = np.argmax(ground_truth, axis=1)

        predictions = np.asarray(predictions).ravel()

        return self.__call__(predictions, ground_truth)


class MeanSquaredError(Metric):
    """
    Metric 1 for regression.
    Class to claculate the mean squared error.
    """
    def __call__(
            self, observations: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        uses numpy to calculate the mean squared error
        """
        return np.mean((observations - ground_truth) ** 2)

    def name(self) -> str:
        """
        Returns:
            str: name of the metric
        """
        return "Mean Squared Error Metric"


class Accuracy(Metric):
    """
    Metric 1 for classification.
    Class to calculate the accuracy.
    """
    def __call__(
            self, observations: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        Finds the accuracy by comparing the ground truth and ground_truth.

        Args:
        observations: np.ndarray
            The observations (x) of the model.
        ground_truth: np.ndarray
            The ground_truths (y) of the model.

        Returns:
            float: accuracy
        """
        return np.mean(observations == ground_truth)

    def name(self) -> str:
        """
        Returns:
            str: name of the metric
        """
        return "Accuracy Metric"


class MeanAbsoluteError(Metric):
    """
    Metric 2 for regression.
    Class to calculate the mean absolute error.
    """
    def __call__(
            self, observations: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        Finds the mean absolute error by getting the mean of the absolute value
        of the difference between ground truth and ground_truth.

        Args:
        observations: np.ndarray
            The observations (x) of the model.
        ground_truth: np.ndarray
            The ground_truths (y) of the model.

        Returns:
            float: mean absolute error
        """
        return np.mean(np.abs(observations - ground_truth))

    def name(self) -> str:
        """
        Returns:
            str: name of the metric
        """
        return "Mean Absolute Error Metric"


class MacroRecall(Metric):
    """
    Metric 2 for classification.
    Class to calculate the macro recall. (recall is binary classification
    metricso we use macro recall to adapt for multi class classification)
    """
    def __call__(
            self, observations: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        Macro average recall in order to be used for more than 2 classes.
        This method calculates the average of each class and then averages.

        Args:
        observations: np.ndarray
            The observations (x) of the model.
        ground_truth: np.ndarray
            The ground_truths (y) of the model.

        Returns:
            float: macro recall
        """
        classes = np.unique(observations)
        recall = []

        for clas in classes:
            true_pos = np.sum((observations == clas) & (ground_truth == clas))
            actual_pos = np.sum(ground_truth == clas)

            if actual_pos == 0:
                recall.append(0.0)
            else:
                recall.append(true_pos / actual_pos)

        return np.mean(recall)

    def name(self) -> str:
        """
        Returns:
            str: name of the metric
        """
        return "Macro Recall Metric"


class Rsquared(Metric):
    """
    Metric 3 for regression.
    Class to calculate the R Squared Metric.
    """
    def __call__(
            self, observations: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        Finds the R Squared Metric value by using the formula
        1 - residual_sum_of_squared / total_sum_of_squared

        Args:
        observations: np.ndarray
            The observations (x) of the model.
        ground_truth: np.ndarray
            The ground_truths (y) of the model.

        Returns:
            float: R Squared
        """
        total_sum_squares = np.sum((observations - np.mean(observations)) ** 2)
        residual_sum_squares = np.sum((observations - ground_truth) ** 2)
        r_squared = 1 - (residual_sum_squares / total_sum_squares)
        return r_squared

    def name(self) -> str:
        """
        Returns:
            str: name of the metric
        """
        return "R Squared Metric"


class Precision(Metric):
    """
    Metric 3 for classification.
    Class to calculate the precision.
    """
    def __call__(
            self, observations: np.ndarray, ground_truth: np.ndarray
            ) -> float:
        """
        Macro average precision in order to be used for more than 2 classes.
        This method calculates the average of each class and then averages.

        Args:
        observations: np.ndarray
            The observations (x) of the model.
        ground_truth: np.ndarray
            The ground_truths (y) of the model.

        Returns:
            float: precision
        """
        classes = np.unique(observations)
        precision = []

        for class_ in classes:
            true_pos = np.sum(
                (observations == class_) & (ground_truth == class_)
                )
            pred_pos = np.sum(ground_truth == class_)

            if pred_pos == 0:
                precision.append(0.0)
            else:
                precision.append(true_pos / pred_pos)

        return np.mean(precision)

    def name(self) -> str:
        """
        Returns:
            str: name of the metric
        """
        return "Precision Metric"
