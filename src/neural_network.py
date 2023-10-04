# neural_network.py
""" Contains class to build and train neural network architectures """

import numpy as np

class NeuralNetwork():
    
    
    def __init__(self, list_architecture:list)->None:
        """Initialization of the architecture of the Neural Network"""
        self.architecture = dict()
        for layer in list_architecture:
            self.architecture[len(self.architecture)] = layer

    def predict(self, input_data:np.ndarray):
        """Predict the output of the architecture passing throughout every layers of the network"""
        result = input_data

        for _, layer in self.architecture.items():
            result = layer.predict(result)

        return result