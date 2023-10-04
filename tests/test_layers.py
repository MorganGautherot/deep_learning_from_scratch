# test_layers.py
""" Test the function inside class of layers.py """

from src.layers import Dense
import numpy as np


def test_dense_sigmoid_initialization() -> None:
    """Test the initialization of a dense layer with
    sigmoid activation function"""
    layers = Dense(neurons_input=5, neurons_output=1, activation="sigmoid")
    assert layers.activation == "sigmoid"


def test_dense_relu_initialization() -> None:
    """Test the initialization of a dense layer with
    relu activation function"""
    layers = Dense(neurons_input=5, neurons_output=1, activation="relu")
    assert layers.activation == "relu"


def test_dense_no_activation_initialization() -> None:
    """Test the initialization of a dense layer with no activation function"""
    layers = Dense(neurons_input=5, neurons_output=1, activation=None)
    assert layers.activation == "linear"


def test_dense_no_arugment_activation_initialization() -> None:
    """Test the initialization of a dense layer with no
    argument for activation function"""
    layers = Dense(neurons_input=5, neurons_output=1)
    assert layers.activation == "linear"


def test_shape_parameters_dense_initialization() -> None:
    """Test the shape of the parameters matrix after layers initializations"""
    layers = Dense(neurons_input=5, neurons_output=5)
    assert layers.parameters.shape == (5, 5)


def test_shape_bias_dense_initialization() -> None:
    """Test the shape of the parameters matrix after layers initializations"""
    layers = Dense(neurons_input=5, neurons_output=5)
    assert layers.bias.shape == (5,)


def test_predict_dense() -> None:
    """Test the predict function of the dense layers"""
    layers = Dense(neurons_input=5, neurons_output=5)
    input_layers = np.array([1, 1, 1, 1, 1])
    predict_result = layers.predict(input_layers)
    assert predict_result.shape == (5,)


def test_relu_activation_function_derivative_dense() -> None:
    """Test the activation_function_derivative function of the
    dense layers for relu activation"""
    # randomness is fixed
    np.random.seed(seed=123)
    # Initialization of the layers
    layers = Dense(neurons_input=5, neurons_output=2, activation="relu")
    input_layers = np.array([1, 1, 1, 1, 1])
    # Compute the excpeted result
    result_layers = np.array([0.03104486, 0.02337508])
    # Predict the input date using the dense layer
    output_layers = layers.predict(input_layers)
    # Compute the activation function derivative
    derivative_result = layers.activation_function_derivative(output_layers)
    approximation_derivative_result = np.around(derivative_result, decimals=8)
    assert np.array_equal(approximation_derivative_result, result_layers)


def test_sigmoid_activation_function_derivative_dense() -> None:
    """Test the activation_function_derivative function of the
    dense layers for sigmoid activation"""
    # randomness is fixed
    np.random.seed(seed=123)
    # Initialization of the layers
    layers = Dense(neurons_input=5, neurons_output=3, activation="sigmoid")
    input_layers = np.array([1, 1, 1, 1, 1])
    # Compute the excpeted result
    result_layers = np.array([0.12688219, 0.12629443, 0.12639507])
    # Predict the input date using the dense layer
    output_layers = layers.predict(input_layers)
    # Compute the activation function derivative
    derivative_result = layers.activation_function_derivative(output_layers)
    approximation_derivative_result = np.around(derivative_result, decimals=8)
    assert np.array_equal(approximation_derivative_result, result_layers)


def test_linear_activation_function_derivative_dense() -> None:
    """Test the activation_function_derivative function of the
    dense layers for linear activation"""
    # Initialization of the layers
    layers = Dense(neurons_input=5, neurons_output=5, activation="linear")
    # Generate input data
    input_layers = np.array([1, 1, 1, 1, 1])
    # Predict the input date using the dense layer
    output_layers = layers.predict(input_layers)
    # Compute the activation function derivative
    derivative_result = layers.activation_function_derivative(output_layers)
    assert np.array_equal(derivative_result, output_layers)
