# Import the packages
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)

from src.neural_network import NeuralNetwork
from src.layers import Dense

# Generate variables
x_linear = np.random.randn(100, 2)
x_linear[:, 1] += 10
x_linear = np.concatenate([np.random.randn(100, 2), x_linear])

# Generate labels
y_linear = np.zeros(100)
y_linear = np.concatenate([np.ones(100), y_linear])
y_linear = y_linear.reshape(-1, 1)

# plot the data
plt.scatter(x_linear[:, 0], x_linear[:, 1], c=y_linear)
plt.show()

# Split the dataset in training and testing subsets
x_linear_train, x_linear_test, y_linear_train, y_linear_test = train_test_split(
    x_linear, y_linear
)

# Data normalization
x_linear_train_norm = (x_linear_train - np.mean(x_linear_train, axis=0)) / (
    np.std(x_linear_train, axis=0)
)
x_linear_test_norm = (x_linear_test - np.mean(x_linear_test, axis=0)) / (
    np.std(x_linear_test, axis=0)
)

# Initialization of the architecture
nn = NeuralNetwork([Dense(2, 1, "sigmoid")])

# Training of the model
nn.fit(x_linear_train_norm, y_linear_train, 10, learning_rate=0.001)

# Visualization of the model
h = 0.02


x_min, x_max = (
    x_linear_train_norm[:, 0].min() - 0.5,
    x_linear_train_norm[:, 0].max() + 0.5,
)
y_min, y_max = (
    x_linear_train_norm[:, 1].min() - 0.5,
    x_linear_train_norm[:, 1].max() + 0.5,
)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
ax = plt.subplot(1, 1, 1)

# Plot the training points
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())

clf = nn

ax = plt.subplot(1, 1, 1)


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)


ax.scatter(
    x_linear_train_norm[:, 0],
    x_linear_train_norm[:, 1],
    c=y_linear_train,
    cmap=cm_bright,
    edgecolors="k",
)

ax.scatter(
    x_linear_train_norm[:, 0],
    x_linear_train_norm[:, 1],
    c=y_linear_train,
    cmap=cm_bright,
    edgecolors="k",
    alpha=0.6,
)

ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xticks(())
ax.set_yticks(())


plt.tight_layout()
plt.show()
