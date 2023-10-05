# performance_metrics.py
""" Contains function to compute the performance of classification model """

import numpy as np


def logloss(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute logloss between y and y_pred"""

    m = y.shape[0]
    return (-1 / m) * np.sum(
        (1 - y) * np.log(1 - y_pred + 1e-6) + y * np.log(y_pred + 1e-6)
    )


def accuracy(y: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute accuracy between y and y_pred"""

    m = y.shape[0]
    y_pred_hard = np.where(y_pred >= threshold, 1, 0)
    return (1 / m) * np.sum(y_pred_hard == y)
