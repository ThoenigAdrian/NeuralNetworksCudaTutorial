import numpy
weights = numpy.array([0.80, 0.87, 0.16, 0.96, 0.89, 0.87, 0.31, 0.08, 0.09, 0.69, 0.03, 0.42])
inputs = numpy.array([0.75,0.98, 0.74, 0.28])
biases = numpy.array([0.68, 0.83, 0.01])

weights = weights.reshape((3, 4))
inputs = inputs.reshape((4, 1))
biases = biases.reshape(-1, 1)

print(weights)

z = numpy.dot(weights, inputs) + biases
print("Z Values: ")
print(z)


def sig(z):
    return 1.0/(1.0+numpy.exp(-z))


print("\nActivations: ")
activations = sig(z)
print(activations)

