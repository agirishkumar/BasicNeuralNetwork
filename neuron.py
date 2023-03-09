import numpy as np

import sampleData

np.random.seed(0)

# X is input feature data/ training dataset
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

X, y = sampleData.spiral_data(100, 3)


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # by defining the shape this way for weights, we don't need to do transpose.
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()
layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)

# layer2 = LayerDense(5, 2)
# layer2.forward(layer1.output)
# print(layer2.output)
