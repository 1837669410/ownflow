import typing
from abc import abstractmethod

import numpy as np

import ownflow as of


class BaseLayer:

    def __init__(self):
        self.name = None
        self.order = None
        self.in_out = {}
        self._x = None

    @abstractmethod
    def forward(self, x: typing.Union[np.ndarray | of.variable.Variable]) -> of.variable.Variable:
        pass

    @abstractmethod
    def backward(self):
        pass

    def _process_input(self, x: typing.Union[np.ndarray | of.variable.Variable]) -> np.ndarray:
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
            x = of.variable.Variable(x)
            x.info['new_layer_order'] = 0 # 标记第一层

        self.order = x.info['new_layer_order']
        self.in_out['in'] = x
        return x.data

    def _warp_out(self, out) -> of.variable.Variable:
        out = of.variable.Variable(out)
        out.info['new_layer_order'] = self.order + 1
        self.in_out['out'] = out
        return out


class ParamLayer(BaseLayer):

    def __init__(self, w_shape, w_init, b_init, activation, use_bias):
        super().__init__()

        self.params = {}
        self.w = np.empty(w_shape, dtype=np.float32)
        self.params['w'] = self.w
        if use_bias:
            self.b = np.empty((1, w_shape[-1]), dtype=np.float32)
            self.params['b'] = self.b
        self.use_bias = use_bias

        if w_init is None:
            self.w[:] = np.random.normal(0., 0.01, self.w.shape)
        else:
            TypeError()

        if b_init is None:
            self.b[:] = np.full(self.b.shape, 0.)
        else:
            TypeError()

        if activation is None:
            self.activation = of.activations.Linear()
        elif isinstance(activation, of.activations.Activation):
            self.activation = activation
        else:
            TypeError()

        self._wx_b = None


class Linear(ParamLayer):

    def __init__(self,
                 in_features,
                 out_features,
                 w_init=None,
                 b_init=None,
                 activation=None,
                 use_bias=True):
        super().__init__(
            w_shape=(in_features, out_features),
            w_init=w_init,
            b_init=b_init,
            activation=activation,
            use_bias=use_bias
        )

    def forward(self, x):
        self._x = self._process_input(x)
        self._wx_b = self._x @ self.w

        if self.use_bias:
            self._wx_b += self.b

        self._activation = self.activation.forward(self._wx_b)
        out = self._warp_out(self._activation)
        return out

    def backward(self):
        delta = self.in_out['out'].error
        delta *= self.activation.backward(self._wx_b)

        grads = {'w': self._x.T @ delta}
        if self.use_bias:
            grads['b'] = np.sum(delta, axis=0, keepdims=True)
        self.in_out['in'].set_error(delta @ self.w.T)
        return grads