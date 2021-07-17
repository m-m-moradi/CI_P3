import numpy as np


class Sigmoid:
    @classmethod
    def function(cls, x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def derivative(cls, x):
        return cls.function(x) * (1 - cls.function(x))


class Relu:
    @classmethod
    def function(cls, x):
        return np.where(x > 0, x, x * 0.01)

    @classmethod
    def derivative(cls, x):
        # return np.greater(x, 0).astype(int)
        y = np.array(x, copy=True)
        y[y <= 0] = 0.01
        y[y > 0] = 1
        return y


class Tanh:
    @classmethod
    def function(cls, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    @classmethod
    def derivative(cls, x):
        return 1 - cls.function(x) ** 2


min_random = -1.5
max_random = 1.5
# Making a List of 2D arrays with respect to network layers(one for each layer) for weights.
# Matrices can be filled with random values or by zeros.
def weight_matrix(layers, random=False):
    mat = [None for i in range(len(layers))]
    for i in range(0, len(layers) - 1, 1):
        CL = layers[i]
        NL = layers[i + 1]
        if random:
            mat[i + 1] = (np.random.random(size=NL * CL) * (max_random - min_random) + min_random).reshape(NL, CL)
        else:
            mat[i + 1] = np.zeros((NL, CL))
    return mat


# Making a List of 2D arrays with respect to network layers(one for each layer) for biases.
# Matrices can be filled with random values or by zeros.
def bias_matrix(layers, random=False):
    mat = [None for i in range(len(layers))]
    for i in range(1, len(layers), 1):
        CL = layers[i]
        if random:
            mat[i] = (np.random.random(size=CL * 1) * (max_random - min_random) + min_random).reshape(CL, 1)
        else:
            mat[i] = np.zeros((CL, 1))
    return mat


class NeuralNetwork():

    def __init__(self, layer_sizes):
        self.layers = layer_sizes
        # Reserving place for matrices.
        # list of 2D arrays, each array is relative to a specific layers.
        self.w = [None for i in range(len(self.layers))]  # weights (None, w1 ... wn)
        self.b = [None for i in range(len(self.layers))]  # biases (None, b1 ... bn)
        self.z = [None for i in range(len(self.layers))]  # neuron summation (before activation function) (None, z1 ... zn)
        self.a = [None for i in range(len(self.layers))]  # activation numbers (a0 ... an)
        self.activation = None

        self.set_weights(weight_matrix(self.layers, random=True))
        self.set_biases(bias_matrix(self.layers, random=True))
        self.set_activation(Sigmoid)

    def _feed(self, sample):
        self.a[0] = sample  # filling the first layer

    def forward(self, x):
        # x example: np.array([[0.1], [0.2], [0.3]])
        self._feed(x)
        for layer in range(1, len(self.layers)):
            self.z[layer] = np.matmul(self.w[layer], self.a[layer - 1]) + self.b[layer]
            self.a[layer] = self.activation.function(self.z[layer])

        return self.a[-1]

    def set_activation(self, _cls):
        self.activation = _cls

    def set_weights(self, w):
        self.w = w

    def set_biases(self, b):
        self.b = b

    def get_output(self):
        return self.a[-1]

