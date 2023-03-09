import sys
import numpy as np
import matplotlib

print("Python version: ", sys.version)
print("Numpy version: ", np.__version__)
print("Matplotlib version: ", matplotlib.__version__)

inputs = [1.0, 2.0, 3.0, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]
# biases = [2, 3, 0.5]
# weights2 = [[0.1, -0.14, 0.5],
#            [-0.5, 0.12, -0.33],
#            [-0.44, 0.73, -0.13]]
# biases2 = [-1, 2, -0.5]

# layer_outputs = []  # output of each layer
# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input * weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)
#
# print(layer_outputs)

# transpose for valid dot product of inputs and weights.
# layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
# print("Layer1 o/p: \n" , layer1_outputs)
# layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
# print("Layer2 o/p: \n" ,layer2_outputs)
