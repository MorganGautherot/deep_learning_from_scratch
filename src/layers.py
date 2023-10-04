# Layers.py
""" Contains all the layers to build neural network architecture """

import numpy as np
from src.activation_functions import relu, sigmoid

class Dense():
    
    def __init__(self, neurons_input:int, neurons_output:int, activation:str=None) -> None:
        """ This function save core information to initialize the parameters and the bias of the layer"""
        self.parameters = np.random.random([neurons_input, neurons_output])*0.01
        self.bias = np.zeros(neurons_output)
        
        if activation == 'sigmoid':
            self.activation = 'sigmoid'
            self.activation_function = sigmoid
        elif activation == 'relu':
            self.activation ='relu'
            self.activation_function = relu
        else :
            self.activation = 'linear'
            self.activation_function = lambda x: x