# Import the packages
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
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

# Generate data
x_moon, y_moon = make_moons(200, noise=0.01, random_state=123)
y_moon = y_moon.reshape(-1, 1)

# plot the data
plt.scatter(x_moon[:, 0], x_moon[:, 1], c=y_moon)
plt.show()

# Split the dataset in training and testing subsets
x_train, x_test, y_train, y_test = train_test_split(x_moon, y_moon)

# Data normalization
x_train_norm = (x_train - np.mean(x_train, axis=0))/(np.std(x_train, axis=0))
x_test_norm = (x_test - np.mean(x_test, axis=0))/(np.std(x_test, axis=0))

# Initialization of the architecture
nn = NeuralNetwork([ Dense(2, 10, 'relu'),
                  Dense(10, 1, 'sigmoid')])

# Training of the model
nn.fit(x_train_norm, y_train, 5000, learning_rate=0.01)

# Visualization of the model
h = 0.02


x_min, x_max = x_train_norm[:, 0].min() - 0.5, x_train_norm[:, 0].max() + 0.5
y_min, y_max = x_train_norm[:, 1].min() - 0.5, x_train_norm[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
ax = plt.subplot(1, 1, 1)

## Plot the training points
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


ax.scatter(x_train_norm[:, 0], x_train_norm[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")

ax.scatter(
    x_train_norm[:, 0],
    x_train_norm[:, 1],
    c=y_train,
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