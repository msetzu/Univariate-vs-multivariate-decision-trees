import numpy
from scipy.spatial.distance import hamming
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from planes.planes import Hyperplane
from trees.structure.trees import InternalNode


def gini(hyperplane: Hyperplane | InternalNode, data: numpy.ndarray, labels: numpy.ndarray,
         classes: numpy.ndarray) -> float:
    """Compute Gini for the given `hyperplane` on the given `data` and `labels`.

    Args:
        hyperplane: The hyperplane whose gini to compute
        data: The data to compute the gini on
        labels: The labels to compute the gini on
        classes: Set of classes

    Returns:
        The gini for the given gini hyperplane.
    """
    if labels.size == 0:
        return 0

    covered_indices = hyperplane(data) if isinstance(hyperplane, Hyperplane) else hyperplane.hyperplane(data)
    noncovered_indices = ~covered_indices
    gini_weights = covered_indices.sum() / labels.size, noncovered_indices.sum() / labels.size

    # gini per split
    covered_squared_purities = [(sum(labels[covered_indices] == c) / covered_indices.sum()) ** 2
                                if covered_indices.sum() > 0 else 0
                                for c in classes]
    noncovered_squared_purities = [(sum(labels[noncovered_indices] == c) / noncovered_indices.sum()) ** 2
                                   if noncovered_indices.sum() > 0 else 0
                                   for c in classes]

    # weighted gini
    covered_gini = (1 - sum(covered_squared_purities))
    noncovered_gini = (1 - sum(noncovered_squared_purities))
    impurity = gini_weights[0] * covered_gini + gini_weights[1] * noncovered_gini

    return impurity


def accuracy(hyperplane: Hyperplane | InternalNode, data: numpy.ndarray, labels: numpy.ndarray) -> float:
    """Compute the accuracy delta between the split (through `hyperplane`) and not split `data`.

    Args:
       hyperplane: The hyperplane whose gini to compute.
       data: The data to compute the gini on.
       labels: The labels to compute the gini on.
       classes: Set of classes.

    Returns:
        The label deviation for the given `hyperplane`.
    """
    return _binary_metric_split(hyperplane, data, labels, "accuracy")


def f1(hyperplane: Hyperplane | InternalNode, data: numpy.ndarray, labels: numpy.ndarray) -> float:
    """Compute the f1 delta between the split (through `hyperplane`) and not split `data`.

    Args:
       hyperplane: The hyperplane whose gini to compute.
       data: The data to compute the gini on.
       labels: The labels to compute the gini on.
       classes: Set of classes.

    Returns:
        The f1 for the given `hyperplane`.
    """
    return _binary_metric_split(hyperplane, data, labels, "f1")


def auc(hyperplane: Hyperplane | InternalNode, data: numpy.ndarray, labels: numpy.ndarray) -> float:
    """Compute the AUC delta between the split (through `hyperplane`) and not split `data`.

    Args:
       hyperplane: The hyperplane whose gini to compute.
       data: The data to compute the gini on.
       labels: The labels to compute the gini on.
       classes: Set of classes.

    Returns:
        The AUC for the given `hyperplane`.
    """
    return _binary_metric_split(hyperplane, data, labels, "auc")


def _binary_metric_split(hyperplane: Hyperplane | InternalNode, data: numpy.ndarray, labels: numpy.ndarray,
                         metric: str) -> float:
    """Compute the accuracy delta between the split (through `hyperplane`) and not split `data`.

    Args:
       hyperplane: The hyperplane whose gini to compute.
       data: The data to compute the gini on.
       labels: The labels to compute the gini on.
       metric: Metric to use.

    Returns:
        The label deviation for the given `hyperplane`.
    """
    if labels.size == 0:
        return 0

    covered_indices = hyperplane(data) if isinstance(hyperplane, Hyperplane) else hyperplane.hyperplane(data)
    noncovered_indices = ~covered_indices
    covered_indices_class = round(labels[covered_indices].mean())
    noncovered_indices_class = round(labels[noncovered_indices].mean())
    split_labels = numpy.full(labels.size, numpy.nan)
    split_labels[covered_indices] = covered_indices_class
    split_labels[noncovered_indices] = noncovered_indices_class

    match metric:
        case "accuracy":
            score_function = accuracy_score
        case "f1":
            score_function = f1_score
        case "auc":
            score_function = roc_auc_score
        case _:
            raise ValueError(f"Unknown metric: {metric}")

    score = score_function(labels, split_labels)

    return score


def label_deviation(hyperplane: Hyperplane, data: numpy.ndarray, labels: numpy.ndarray) -> float:
    """Compute the delta in standard deviation of the labels between the split (through `hyperplane`) and
    not split `data`.

     Args:
        hyperplane: The hyperplane whose gini to compute.
        data: The data to compute the gini on.
        labels: The labels to compute the gini on.
        classes: Set of classes.

    Returns:
        The label deviation for the given `hyperplane`.
    """
    std_labels = labels.std()
    node_size = data.shape[0]
    left_idx = hyperplane(data)
    left_child_labels, right_child_labels = labels[left_idx], labels[~left_idx]
    weighted_children_errors = [left_child_labels.std() * left_child_labels.size,
                                right_child_labels.std() * right_child_labels.size]
    # correct for no labels
    weighted_children_errors[0] = weighted_children_errors[0] if left_child_labels.size > 0 else 0
    weighted_children_errors[1] = weighted_children_errors[1] if right_child_labels.size > 0 else 0

    return std_labels - (1 / node_size) * sum(weighted_children_errors)
