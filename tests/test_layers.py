# test_layers.py
""" Test the function inside class of layers.py """

from src.layers import Dense
import numpy as np

# Unit test for the class Dense
## Unit test for the initialization function

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

## Unit test for the predict function

def test_predict_dense() -> None:
    """Test the predict function of the dense layers"""
    layers = Dense(neurons_input=5, neurons_output=5)
    input_layers = np.array([1, 1, 1, 1, 1])
    predict_result = layers.predict(input_layers)
    assert predict_result.shape == (5,)

## Unit test for the activation_function_derivative function

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

## Unit test for the compute_gradient function

def test_compute_grad() -> None:
    """Test the compute_grad function for dense layer"""
    # randomness is fixed
    np.random.seed(seed=123)
    # Initialization of the layers
    layers = Dense(neurons_input=5, neurons_output=3, activation="sigmoid")
    input_layers = np.array([1, 1, 1, 1, 1]).reshape(1, 5)
    # Compute the excpeted result
    result_layers = np.array([[0.0015318 , 0.00214296, 0.00271719, 0.00185242, 0.00113495]])
    # Predict the input date using the dense layer
    output_layers = layers.predict(input_layers)
    # Compute the activation function derivative
    derivative_result = layers.compute_grad(output_layers)
    approximation_derivative_result = np.around(derivative_result, decimals=8)
    assert np.array_equal(approximation_derivative_result, result_layers)

## Unit test for the parameters_update function

def test_parameters_update() -> None:
    """Test the parameters_update function for dense layer"""
    # randomness is fixed
    np.random.seed(seed=123)
    # Initialization of the layers
    layers = Dense(neurons_input=2, neurons_output=1, activation="sigmoid")
    input_layers = np.array([1, 1]).reshape(1, 2)
    # Compute the excpeted result
    result_layers = np.array([[-0.05614942], [-0.06025272]])
    # Predict the input date using the dense layer
    output_layers = layers.predict(input_layers)
    # Compute the activation function derivative
    _ = layers.compute_grad(output_layers)
    # update the parameters
    _ = layers.parameters_update(output_layers)

    approximation_parameters = np.around(layers.parameters, decimals=8)
    assert np.array_equal(approximation_parameters, result_layers)

