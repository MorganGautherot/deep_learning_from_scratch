# test_performance_metrics.py
""" Test the function inside performance_metrics.py """

from src.performance_metrics import logloss, accuracy
import numpy as np


def test_logloss() -> None:
    """Test the result of logloss function"""
    true_result = np.array([0, 1, 1, 0])
    prediction_result = np.array([0.001, 0.23, 0.93, 0.1])
    # Compute logloss
    logloss_computed = logloss(true_result, prediction_result)
    logloss_approximation = np.around(logloss_computed, decimals=3)
    assert logloss_approximation == 0.412


def test_accuracy() -> None:
    """Test the result of logloss function"""
    true_result = np.array([0, 1, 1, 0])
    prediction_result = np.array([0, 0, 1, 0])
    # Compute accuracy
    accuracy_computed = accuracy(true_result, prediction_result)
    assert accuracy_computed == 0.75
