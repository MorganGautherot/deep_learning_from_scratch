# test_activation_functions.py
""" Test the function inside activation_functions.py"""

from src.activation_functions import relu, sigmoid
import numpy as np


def test_negative_entry_relu() -> None:
    """Test that the function return 0 when the entry is negative"""
    relu_result = relu(-1)
    assert 0 == relu_result


def test_positive_entry_relu() -> None:
    """Test that the function return 0 when the entry is positive"""
    test_value = 5
    relu_result = relu(test_value)
    assert test_value == relu_result


def test_zero_entry_relu() -> None:
    """Test that the function return 0 when the entry is 0"""
    relu_result = relu(0)
    assert 0 == relu_result


def test_negative_entry_sigmoid() -> None:
    """Test that the function return 0 when the entry is negative"""
    sigmoid_result = sigmoid(-1)
    sigmoid_approximate_result = np.around(sigmoid_result, decimals=5)
    assert 0.26894 == sigmoid_approximate_result


def test_positive_entry_sigmoid() -> None:
    """Test that the function return 0 when the entry is positive"""
    sigmoid_result = sigmoid(100)
    assert 1.0 == sigmoid_result


def test_zero_entry_sigmoid() -> None:
    """Test that the function return 0 when the entry is 0"""
    sigmoid_result = sigmoid(0)
    assert 0.5 == sigmoid_result
