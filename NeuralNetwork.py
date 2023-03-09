import numpy as np


class NeuralNetwork:
    # input_size refers to the number of input features or dimensions in the input data that the neural network will
    # process.
    # hidden_sizes is a list that contains the number of neurons in each hidden layer of a neural network.
    # output_size refers to the number of neurons in the output layer of the network.
    def __init__(self, input_size, hidden_sizes, output_size):
        self.weights = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.weights.append(np.random.randn(prev_size, hidden_size))
            prev_size = hidden_size
        self.weights.append(np.random.randn(prev_size, output_size))

    def forward(self, x):
        hidden_layer = x
        for weight in self.weights[:-1]:
            hidden_layer = np.dot(hidden_layer, weight)
            hidden_layer = np.maximum(0, hidden_layer)  # ReLU activation function
        output_layer = np.dot(hidden_layer, self.weights[-1])
        return output_layer

    def train(self, x, y, learning_rate):
        # x is the input data.
        # Perform a forward pass through the network.
        # y is the output data.
        hidden_layers = [x]
        output_layer = self.forward(x)

        # Compute the error and update the weights
        error = y - output_layer
        delta = error
        for i in range(len(hidden_layers) - 1, -1, -1):
            gradient = np.dot(hidden_layers[i].T, delta) * learning_rate
            self.weights[i] += gradient
            delta = np.dot(delta, self.weights[i].T)
            delta[hidden_layers[i] <= 0] = 0  # Backpropagation through ReLU activation

        return error
