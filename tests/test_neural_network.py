# test_neural_network.py
""" Test the function inside class of neural_network.py """

from src.neural_network import NeuralNetwork
from src.layers import Dense
import numpy as np

# Unit test for the class NeuralNetwork
## Unit test for the initialization function

def test_neural_network_initialization()->None:
    """Test the initialization of an object NeuralNetwork"""
    model = NeuralNetwork([Dense(2, 20, 'relu'), 
                              Dense(20, 10, 'relu'), 
                              Dense(10, 1, 'sigmoid')])
    assert model.architecture[0].parameters.shape == (2, 20)
    assert model.architecture[1].parameters.shape == (20, 10)
    assert model.architecture[2].parameters.shape == (10, 1)

## Unit test for the predict function

def test_predict()->None:
    """Test the predict throughout the network"""
    # Initialization of the network
    model = NeuralNetwork([Dense(2, 20, 'relu'), 
                              Dense(20, 10, 'relu'), 
                              Dense(10, 5, 'sigmoid')])
    # Generation of the input data
    input_data = np.array([[2, 2], [2, 2]])
    # output of the network 
    output_data = model.predict(input_data)
    assert output_data.shape == (2, 5)