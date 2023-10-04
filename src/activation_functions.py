# function.py
""" Contains all the activation functions """

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Perform ReLU function"""
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Perform sigmoid function"""
    return 1 / (1 + np.exp(-x))
