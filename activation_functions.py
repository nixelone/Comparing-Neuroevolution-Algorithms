import numpy as np


def abs_(x):
    return np.abs(x)


def cube(x):
    return x ** 3


def exp(x):
    return np.exp(x)


def identity(x):
    return x


def relu(x):
    return np.maximum(0, x)


def log(x):
    return np.log(x)


def sigmoid(x):
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


def sin(x):
    return np.sin(x)


def softplus(x):
    return np.log(1 + np.exp(x))


def square(x):
    return x ** 2


def tanh(x):
    return np.tanh(x)


functions = {
    'abs': abs_,
    'cube': cube,
    'exp': exp,
    'identity': identity,
    'relu': relu,
    'log': log,
    'sigmoid': sigmoid,
    'sin': sin,
    'softplus': softplus,
    'square': square,
    'tanh': tanh
}
