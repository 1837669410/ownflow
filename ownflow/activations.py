import numpy as np


class Activation:

    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, x):
        pass


class Linear(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def backward(self, x):
        return np.ones_like(x)


class Relu(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))