# test_neural_network.py
""" Test the function inside class of test_neural_network.py """

# Unit test for the class NeuralNetwork
## Unit test for the initialization function

from src.neural_network import NeuralNetwork
from src.layers import Dense

def test_neural_network_initialization()->None:
    """Test the initialization of an object NeuralNetwork"""
    nn_object = NeuralNetwork([Dense(2, 20, 'relu'), 
                              Dense(20, 10, 'relu'), 
                              Dense(10, 1, 'sigmoid')])
    assert nn_object.architecture[0].parameters.shape == (2, 20)
    assert nn_object.architecture[1].parameters.shape == (20, 10)
    assert nn_object.architecture[2].parameters.shape == (10, 1)