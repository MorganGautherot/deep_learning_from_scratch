# Layers.py
""" Contains all the layers to build neural network architecture """

import numpy as np
from src.activation_functions import relu, sigmoid


class Dense:
    def __init__(
        self, neurons_input: int, neurons_output: int, activation: str = None
    ) -> None:
        """This function save core information to initialize the parameters and
        the bias of the layer"""
        self.parameters = np.random.random([neurons_input, neurons_output]) * 0.01
        self.bias = np.zeros(neurons_output)

        if activation == "sigmoid":
            self.activation = "sigmoid"
            self.activation_function = sigmoid
        elif activation == "relu":
            self.activation = "relu"
            self.activation_function = relu
        else:
            self.activation = "linear"
            self.activation_function = lambda x: x

    def predict(self, a_prev: np.ndarray) -> np.ndarray:
        """This function compute a matrix multiplication between a_prev and
        the parameters of the model"""

        self.a_prev = a_prev

        self.z = np.dot(self.a_prev, self.parameters) + self.bias

        self.a = self.activation_function(self.z)

        return self.a

    def activation_function_derivative(self, da: np.array) -> np.ndarray:
        """Compute de deritave of the activation function"""

        if self.activation == "relu":
            dz = da
            dz[self.z <= 0] = 0

        elif self.activation == "sigmoid":
            s = 1 / (1 + np.exp(-self.z))
            dz = da * s * (1 - s)

        else:
            dz = da

        return dz

    def compute_grad(self, da: np.ndarray) -> np.ndarray:
        """Compute gradient for every parameters of this layer"""

        dz = self.activation_function_derivative(da)

        m = self.a.shape[1]

        self.dw = (1 / m) * np.dot(self.a_prev.T, dz)
        self.db = (1 / m) * np.sum(dz, axis=0)

        da_prev = np.dot(dz, self.parameters.T)

        return da_prev

    def parameters_update(self, learning_rate: float) -> None:
        """Update the values of the parameters using gradient
        descent algorithm"""

        self.parameters = self.parameters - learning_rate * self.dw
        self.bias = self.bias - learning_rate * self.db
