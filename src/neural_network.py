# neural_network.py
""" Contains class to build and train neural network architectures """

import numpy as np
from src.performance_metrics import logloss, accuracy

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
    
    def compute_grad(self, y_train:np.ndarray, y_pred:np.ndarray)->None:
        """Compute gradient for every layers of the network"""
        da_prev = - (y_train/(y_pred+0.0000001)) + (1-y_train) / (1 - y_pred+0.0000001)
        
        for _, layer in reversed(self.architecture.items()):
            da_prev = layer.compute_grad(da_prev)