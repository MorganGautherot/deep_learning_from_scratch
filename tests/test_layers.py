# test_layers.py
""" Test the function inside class of layers.py """

from src.layers import Dense

def test_dense_sigmoid_initialization() -> None:
    """ Test the initialization of a dense layer with sigmoid activation function"""
    layers = Dense(neurons_input=5, 
                   neurons_output=1,
                   activation='sigmoid')
    assert layers.activation == 'sigmoid'

def test_dense_relu_initialization() -> None:
    """ Test the initialization of a dense layer with relu activation function"""
    layers = Dense(neurons_input=5, 
                   neurons_output=1,
                   activation='relu')
    assert layers.activation == 'relu'

def test_dense_no_activation_initialization() -> None:
    """ Test the initialization of a dense layer with no activation function"""
    layers = Dense(neurons_input=5, 
                   neurons_output=1,
                   activation=None)
    assert layers.activation == 'linear'

def test_dense_no_arugment_activation_initialization() -> None:
    """ Test the initialization of a dense layer with no argument for activation function"""
    layers = Dense(neurons_input=5, 
                   neurons_output=1)
    assert layers.activation == 'linear'

def test_shape_parameters_dense_initialization() -> None:
    """ Test the shape of the parameters matrix after layers initializations"""
    layers = Dense(neurons_input=5, 
                   neurons_output=5)
    assert layers.parameters.shape == (5, 5)

def test_shape_bias_dense_initialization() -> None:
    """ Test the shape of the parameters matrix after layers initializations"""
    layers = Dense(neurons_input=5, 
                   neurons_output=5)
    assert layers.bias.shape == (5,)

    