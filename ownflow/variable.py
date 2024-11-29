import numpy as np


class Variable:

    def __init__(self, value):
        self.data = value
        self._error = np.empty_like(value, dtype=np.float32)
        self.info = {}

    def set_error(self, error):
        self._error = error

    @property
    def error(self):
        return self._error

    @property
    def shape(self):
        return self.data.shape