import math

import numpy as np


class Initializer:

    def init(self, x):
        pass


class Uniform_(Initializer):

    def __init__(self, a=0.0, b=1.0):
        super().__init__()

        self._a = a
        self._b = b

    def init(self, x):
        x[:] = np.random.uniform(self._a, self._b, x.shape)


class Normal_(Initializer):

    def __init__(self, mean=0.0, std=1.0):
        super().__init__()

        self._mean = mean
        self._std = std

    def init(self, x):
        x[:] = np.random.normal(self._mean, self._std, x.shape)


class Constant_(Initializer):

    def __init__(self, value):
        super().__init__()

        self._value = value

    def init(self, x):
        x[:] = np.full(x.shape, self._value)


class Ones_(Initializer):

    def __init__(self):
        super().__init__()

    def init(self, x):
        x[:] = np.ones(x.shape)


class Zeros_(Initializer):

    def __init__(self):
        super().__init__()

    def init(self, x):
        x[:] = np.zeros(x.shape)


class Eye_(Initializer):

    def __init__(self):
        super().__init__()

    def init(self, x):
        shape = x.shape
        if len(shape) == 2 and shape[0] == shape[1]:
            x[:] = np.eye(shape[0])
        else:
            raise ValueError('x必须是方阵')


def _cal_fan_in_and_fan_out(x):
    fan_in, fan_out = x.shape[0], x.shape[1]
    if len(x) > 2:
        for s in x.shape[2:]:
            fan_in *= s
            fan_out *= s
    return fan_in, fan_out


class Xavier_Uniform_(Initializer):

    def __init__(self, gain=1.0):
        super().__init__()

        self._gain = gain

    def init(self, x):
        fan_in, fan_out = _cal_fan_in_and_fan_out(x)
        std = self._gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        x[:] = np.random.uniform(-a, a, x.shape)


class Xavier_Normal_(Initializer):

    def __init__(self, gain=1.0):
        super().__init__()

        self._gain = gain

    def init(self, x):
        fan_in, fan_out = _cal_fan_in_and_fan_out(x)
        std = self._gain * math.sqrt(2.0 / (fan_in + fan_out))
        x[:] = np.random.normal(0.0, std, x.shape)


def _cal_gain(nonlinearity, a=None):
    if nonlinearity == 'linear' or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if a is None:
            a = 0.01
        else:
            a = a
        return math.sqrt(2.0 / (1 + a ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        ValueError('不支持该激活函数')


class Kaiming_Uniform_(Initializer):

    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super().__init__()

        self._a = a
        self._mode = mode
        self._nonlinearity = nonlinearity

    def init(self, x):
        fan_in, fan_out = _cal_fan_in_and_fan_out(x)
        if self._mode == 'fan_in':
            fan = fan_in
        else:
            fan = fan_out
        gain = _cal_gain(self._nonlinearity, self._a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        x[:] = np.random.uniform(-bound, bound, x.shape)


class Kaiming_Normal_(Initializer):

    def __init__(self, a=0, mode='fan_in', nonlinearity='leaky_relu'):
        super().__init__()

        self._a = a
        self._mode = mode
        self._nonlinearity = nonlinearity

    def init(self, x):
        fan_in, fan_out = _cal_fan_in_and_fan_out(x)
        if self._mode == 'fan_in':
            fan = fan_in
        else:
            fan = fan_out
        gain = _cal_gain(self._nonlinearity, self._a)
        std = gain / math.sqrt(fan)
        x[:] = np.random.normal(0.0, std, x.shape)


class Trunc_Normal_(Initializer):

    def __init__(self, mean=0.0, std=1.0, a=None, b=None):
        super().__init__()

        self._mean = mean
        self._std = std
        self._a = a
        self._b = b

    def init(self, x):
        x[:] = np.random.normal(self._mean, self._std, x.shape)
        if self._a is None or self._b is None:
            value = self._mean + self._std * 2
            x[:] = np.clip(x, -value, value)
        else:
            x[:] = np.clip(x, self._a, self._b)


class Orthogonal_(Initializer):

    def __init__(self, gain=1.0):
        super().__init__()

        self._gain = gain

    def init(self, x):
        row, col = x.shape

        xv = np.random.normal(0.0, 1.0, x.shape)
        if row < col:
            xv = xv.T

        q, r = np.linalg.qr(xv)
        d = np.diag(r, 0)
        ph = np.sign(d)
        q *= ph

        if row < col:
            q = q.T

        x[:] = q * self._gain


class Sparse_(Initializer):

    def __init__(self, sparsity, std=0.01):
        super().__init__()

        self._sparsity = sparsity
        self._std = std

    def init(self, x):
        row, col = x.shape
        num_zeros = int(math.ceil(self._sparsity * row))

        xv = np.random.normal(0., self._std, x.shape)
        for col_idx in range(col):
            row_indices = np.random.permutation(row)
            zeros_indices = row_indices[:num_zeros]
            xv[zeros_indices, col_idx] = 0
            x[:] = xv
