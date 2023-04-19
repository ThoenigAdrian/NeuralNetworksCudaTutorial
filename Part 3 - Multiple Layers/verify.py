import numpy
import numpy as np

weights = numpy.array([1.62, -0.61, -0.53, -1.07, 0.87, -2.30, 1.74, -0.76, 0.32, -0.25, 1.46, -2.06, -0.32, -0.38, 1.13,
                       -1.10, -0.17, -0.88, 0.04, 0.58, -1.10, 1.14, 0.90, 0.50, 0.90, -0.68, -0.12, -0.94, -0.27, 0.53,
                       -0.69, -0.40, -0.69, -0.85, -0.67, -0.01, -1.12, 0.23, 1.66, 0.74, -0.19, -0.89, -0.75, 1.69, 0.05,
                       -0.64, 0.19, 2.10, 0.12, 0.62, 0.30, -0.35, -1.14, -0.35, -0.21, 0.59, 0.84, 0.93, 0.29, 0.89, -0.75,
                       1.25, 0.51, -0.30, 0.49, -0.08, 1.13, 1.52, 2.19, -1.40, -1.44, -0.50, 0.16, 0.88, 0.32, -2.02])
biases = numpy.array([-0.31, 0.83, 0.23, 0.76, -0.22, -0.20, 0.19, 0.41, 0.20, 0.12, -0.67])
activations = np.zeros(19)
activations[:8] = np.array([0.38, 0.12, 1.13, 1.20, 0.19, -0.38, -0.64, 0.42])
activations = [activations[:8].reshape(-1, 1), activations[8:14].reshape(-1, 1), activations[14:18].reshape(-1, 1), activations[18:].reshape(-1, 1)]
weights = [weights[:48].reshape(6, 8), weights[48:72].reshape(4, 6), weights[72:].reshape(1, 4)]
biases = [biases[:6].reshape(-1, 1), biases[6:10].reshape(-1, 1), biases[10:].reshape(-1, 1)]

def sig(z):
    return 1.0/(1.0+numpy.exp(-z))

z_values = []
for layer_index in range(3):
  z = numpy.dot(weights[layer_index], activations[layer_index]) + biases[layer_index]
  z_values.append(z)
  activations[layer_index + 1] += sig(z)


print("Z Values: ")
for layer_index in range(3):
  print(z_values[layer_index])

print("Activations: ")
for layer_index in range(1, 4):
  print(activations[layer_index])