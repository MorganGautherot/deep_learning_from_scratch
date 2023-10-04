# neural_network.py
""" Contains class to build and train neural network architectures """


class NeuralNetwork():
    
    
    def __init__(self, list_architecture:list)->None:
        """Initialization of the architecture of the Neural Network"""
        self.architecture = dict()
        for layer in list_architecture:
            self.architecture[len(self.architecture)] = layer

    