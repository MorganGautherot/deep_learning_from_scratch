# test_neural_network.py
""" Test the function inside class of neural_network.py """

from src.neural_network import NeuralNetwork
from src.performance_metrics import logloss, accuracy
from src.layers import Dense
import numpy as np

# Unit test for the class NeuralNetwork
# Unit test for the initialization function


def test_neural_network_initialization() -> None:
    """Test the initialization of an object NeuralNetwork"""
    model = NeuralNetwork(
        [Dense(2, 20, "relu"), Dense(20, 10, "relu"), Dense(10, 1, "sigmoid")]
    )
    assert model.architecture[0].parameters.shape == (2, 20)
    assert model.architecture[1].parameters.shape == (20, 10)
    assert model.architecture[2].parameters.shape == (10, 1)


# Unit test for the predict function


def test_predict() -> None:
    """Test the predict throughout the network"""
    # Initialization of the network
    model = NeuralNetwork(
        [Dense(2, 20, "relu"), Dense(20, 10, "relu"), Dense(10, 5, "sigmoid")]
    )
    # Generation of the input data
    input_data = np.array([[2, 2], [2, 2]])
    # output of the network
    output_data = model.predict(input_data)
    assert output_data.shape == (2, 5)


# Unit test for the compute gradient function


def test_compute_grad() -> None:
    """Test the compute of gradient in the network for every layers"""
    # Initialization of the network
    model = NeuralNetwork(
        [Dense(2, 20, "relu"), Dense(20, 10, "relu"), Dense(10, 1, "sigmoid")]
    )
    # Generation of the input data
    input_data = np.array([[0, 0], [10, 10]])
    # Generation of the true data labels
    true_label = np.array([[0], [1]])
    # output of the network
    predict_label = model.predict(input_data)
    # compute gradient for every layers
    model.compute_grad(true_label, predict_label)
    assert model.architecture[0].dw.shape == (2, 20)
    assert model.architecture[0].db.shape == (20,)
    assert model.architecture[1].dw.shape == (20, 10)
    assert model.architecture[1].db.shape == (10,)
    assert model.architecture[2].dw.shape == (10, 1)
    assert model.architecture[2].db.shape == (1,)


# Unit test for the parameters update function


def test_parameters_update() -> None:
    """Test the update of every parameters of the network"""
    # randomness is fixed
    np.random.seed(seed=123)
    # Initialization of the network
    model = NeuralNetwork(
        [Dense(2, 20, "relu"), Dense(20, 10, "relu"), Dense(10, 1, "sigmoid")]
    )

    # Generation of the input data
    input_data = np.array([[0, 0], [2, 2], [5, 5], [10, 10]])
    # Generation of the true data labels
    true_label = np.array([[0], [0], [1], [1]])

    # predict before update
    predict_label_before = model.predict(input_data)

    # Compute the loss before update the parameters
    loss_before_parameters_update = logloss(true_label, predict_label_before)

    # compute gradient for every layers
    model.compute_grad(true_label, predict_label_before)

    # update the parameters for every layers
    model.parameters_update(learning_rate=0.1)

    # predict after update
    predict_label_after = model.predict(input_data)

    # Compute the loss after update the parameters
    loss_after_parameters_update = logloss(true_label, predict_label_after)

    assert loss_before_parameters_update > loss_after_parameters_update


def test_fit_neural_network() -> None:
    """Test the train of a neural network model"""

    # Generate explanatory variable
    x_linear = np.random.randn(100, 2)
    x_linear[:, 1] += 10
    x_linear = np.concatenate([np.random.randn(100, 2), x_linear])

    # Generate labels
    y_linear = np.zeros(100)
    y_linear = np.concatenate([np.ones(100), y_linear])
    y_linear = y_linear.reshape(-1, 1)

    # Data normalization
    x_train_norm = (x_linear - np.mean(x_linear, axis=0)) / (np.std(x_linear, axis=0))

    model = NeuralNetwork([Dense(2, 1, "sigmoid")])

    model.fit(x_train_norm, y_linear, 10, learning_rate=0.001)

    y_pred = model.predict(x_train_norm)

    model_accuracy = accuracy(y_linear, y_pred)

    assert 1 == model_accuracy
