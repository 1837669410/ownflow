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


class Threshold(Activation):

    def __init__(self, threshold, value):
        super().__init__()
        self._threshold = threshold
        self._value = value

    def forward(self, x):
        return np.where(x > self._threshold, x, self._value)

    def backward(self, x):
        return np.where(x > self._threshold, np.ones_like(x), np.zeros_like(x))


class Relu(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, x):
        return np.where(x > 0, np.ones_like(x), np.zeros_like(x))


# Todo 当前RRelu只实现了训练时，因为测试的时候需要保持alpha不变
class RRelu(Activation):

    def __init__(self, low = 1.0 / 8, high = 1.0 / 3):
        super().__init__()

        self._low = low
        self._high = high
        self._alpha = None

    def forward(self, x):
        self._alpha = np.random.uniform(self._low, self._high, x.shape)
        return np.where(x > 0, x, self._alpha * x)

    def backward(self, x):
        return np.where(x > 0, np.ones_like(x), self._alpha)


class HardTanh(Activation):

    def __init__(self, min_val=-1.0, max_val=1.0):
        super().__init__()

        self._min_val = min_val
        self._max_val = max_val

    def forward(self, x):
        return np.clip(x, self._min_val, self._max_val)

    def backward(self, x):
        return np.where(x > self._min_val, np.where(x < self._max_val, np.ones_like(x), np.zeros_like(x)), np.zeros_like(x))


class Relu6(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.clip(x, 0, 6)

    def backward(self, x):
        return np.where(x > 0, np.where(x < 6, np.ones_like(x), np.zeros_like(x)), np.zeros_like(x))


class Sigmoid(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, x):
        return np.exp(-x) / (1.0 + np.exp(-x)) ** 2


class HardSigmoid(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.where(x > -3, np.where(x < 3, x / 6 + 0.5, np.ones_like(x)), np.zeros_like(x))

    def backward(self, x):
        return np.where(x > -3, np.where(x < 3, 1.0 / 6, np.zeros_like(x)), np.zeros_like(x))


class Tanh(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1.0 - np.tanh(x) ** 2


class Silu(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (1.0 + np.exp(-x))

    def backward(self, x):
        left_term = 1 / (1.0 + np.exp(-x))
        right_term = x * np.exp(-x) / (1.0 + np.exp(-x)) ** 2
        return left_term + right_term


class Mish(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * np.tanh(np.log(1 + np.exp(x)))

    def backward(self, x):
        tanh_term = np.tanh(np.log(1 + np.exp(x)))
        left_term = tanh_term
        right_term = x * (1 - tanh_term ** 2) / (1 + np.exp(x)) * np.exp(x)
        return left_term + right_term


class HardSwish(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.where(x > -3, np.where(x < 3, x * (x + 3) / 6, x), np.zeros_like(x))

    def backward(self, x):
        return np.where(x > -3, np.where(x < 3, 1 / 6 * (2 * x + 3), np.ones_like(x)), np.zeros_like(x))


class Elu(Activation):

    def __init__(self, alpha=1.0):
        super().__init__()

        self._alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self._alpha * (np.exp(x) - 1))

    def backward(self, x):
        return np.where(x > 0, np.ones_like(x), self._alpha * np.exp(x))


class Celu(Activation):

    def __init__(self, alpha=1.0):
        super().__init__()

        self._alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self._alpha * (np.exp(x / self._alpha) - 1))

    def backward(self, x):
        return np.where(x > 0, np.ones_like(x), np.exp(x / self._alpha))


class Selu(Activation):

    def __init__(self, alpha=1.67326, scale=1.0507):
        super().__init__()

        self._alpha = alpha
        self._scale = scale

    def forward(self, x):
        return np.where(x > 0, self._scale * x, self._scale * self._alpha * (np.exp(x) - 1))

    def backward(self, x):
        return np.where(x > 0, self._scale, self._scale * self._alpha * np.exp(x))


class Gelu(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def backward(self, x):
        sqrt_term = np.sqrt(2 / np.pi)
        tanh_term = np.tanh(sqrt_term * (x + 0.044715 * x ** 3))

        left_term = 0.5 * (1 + tanh_term)
        right_term = 0.5 * x * (1 - tanh_term ** 2) * sqrt_term * (1 + 3 * 0.044715 * x ** 2)
        return left_term + right_term


class HandShrink(Activation):

    def __init__(self, lambd=0.5):
        super().__init__()

        self._lambd = lambd

    def forward(self, x):
        return np.where(x > -self._lambd, np.where(x < self._lambd, 0, x), x)

    def backward(self, x):
        return np.where(x > -self._lambd, np.where(x < self._lambd, np.zeros_like(x), np.ones_like(x)), np.ones_like(x))


class LeakyRelu(Activation):

    def __init__(self, alpha=0.01):
        super().__init__()

        self._alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self._alpha * x)

    def backward(self, x):
        return np.where(x > 0, np.ones_like(x), self._alpha)


class LogSigmoid(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.log(1 / (1 + np.exp(-x)))

    def backward(self, x):
        return np.exp(-x) / (1 + np.exp(-x))


class Softplus(Activation):

    def __init__(self, beta=1.0, threshold=20):
        super().__init__()

        self._beta = beta
        self._threshold = threshold

    def forward(self, x):
        return np.where(x * self._beta > self._threshold, x, (1 / self._beta) * np.log(1 + np.exp(self._beta * x)))

    def backward(self, x):
        return np.where(x * self._beta > self._threshold, np.ones_like(x), np.exp(self._beta * x) / (1 + np.exp(self._beta * x)))


class SoftShrink(Activation):

    def __init__(self, lambd=0.5):
        super().__init__()

        self._lambd = lambd

    def forward(self, x):
        return np.where(x > -self._lambd, np.where(x < self._lambd, 0, x - self._lambd), x + self._lambd)

    def backward(self, x):
        return np.where(x > -self._lambd, np.where(x < self._lambd, np.zeros_like(x), np.ones_like(x)), np.ones_like(x))


class SoftSign(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.where(x > 0, x / (1 + x), x / (1 - x))

    def backward(self, x):
        return np.where(x > 0, 1 / (1 + x) ** 2, 1 / (1 - x) ** 2)


class TanhShrink(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x - np.tanh(x)

    def backward(self, x):
        return np.tanh(x) ** 2