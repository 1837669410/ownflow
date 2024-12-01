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
            shape = [1] * len(w_shape)
            shape[-1] = w_shape[-1]
            self.b = np.empty(shape, dtype=np.float32)
            self.params['b'] = self.b
        self.use_bias = use_bias

        if w_init is None:
            self.w[:] = np.random.normal(0., 0.01, self.w.shape)
        elif isinstance(w_init, of.initializers.Initializer):
            w_init.init(self.w)
        else:
            TypeError()

        if use_bias:
            if b_init is None:
                self.b[:] = np.full(self.b.shape, 0.)
            elif isinstance(b_init, of.initializers.Initializer):
                b_init.init(self.b)
            else:
                TypeError()

        if activation is None:
            self.activation = of.activations.Linear()
        elif isinstance(activation, of.activations.Activation):
            self.activation = activation
        else:
            TypeError()

        self._wx_b = None
        self._activation = None


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


class Conv2D(ParamLayer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=(3, 3),
                 stride=1,
                 padding=0,
                 w_init=None,
                 b_init=None,
                 activation=None,
                 use_bias=True
                 ):
        super().__init__(
            w_shape=(in_channels,) + kernel_size + (out_channels,),
            w_init=w_init,
            b_init=b_init,
            activation=activation,
            use_bias=use_bias
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        self.pad_img = None

    def forward(self, x):
        # x: [b, c, h, w]
        _x = self._process_input(x)

        # pad
        self.pad_img, tmp_conv = get_pad_and_get_tmp(_x, self.kernel_size, self.stride, self.out_channels, self.padding)

        # conv
        self._wx_b = self.convolution(self.pad_img, self.w, tmp_conv)
        if self.use_bias:
            self._wx_b += self.b.transpose(0, 3, 1, 2)

        self._activation = self.activation.forward(self._wx_b)
        out = self._warp_out(self._activation)
        return out

    def backward(self):
        delta = self.in_out['out'].error
        delta *= self.activation.backward(self._wx_b)

        # dw, db
        dw = np.empty_like(self.w) # [in_c, h, w, out_c]
        # pad_img [b, in_c, h, w] -> [in_c, b, h, w]
        # delta [b, out_c, h, w] -> [b, h, w, out_c]
        # dw [in_c, h, w, out_c] -> [in_c, out_c, h, w]
        dw = self.convolution(self.pad_img.transpose(1, 0, 2, 3), delta.transpose(0, 2, 3, 1), dw.transpose(0, 3, 1, 2))
        grads = {'w': dw.transpose(0, 2, 3, 1)}
        if self.use_bias:
            # delta [b, out_c, h, w] -> [1, out_c, 1, 1] -> [1, 1, 1, out_c]
            grads['b'] = np.sum(delta, axis=(0, 2, 3), keepdims=True).transpose(0, 2, 3, 1)

        # dx
        pad_dx = np.zeros_like(self.pad_img) # [b, in_c, h, w]
        sh, sw, fh, fw = self.stride + self.kernel_size
        # [in_c, h, w, out_c] -> [out_c, h, w, in_c]
        f_t = self.w.transpose(3, 1, 2, 0)
        for i in range(delta.shape[2]):
            for j in range(delta.shape[3]):
                # [b, out_c] @ [out_c, fh*fw*in_c] -> [b, fh*fw*in_c] -> [b, in_c, fh, fw]
                pad_dx[:, :, i*sh: i*sh+fh, j*sw: j*sw+fw] += (delta[:, :, i, j].reshape(-1, self.out_channels) @
                                                               f_t.reshape(self.out_channels, -1)
                                                               ).reshape(-1, pad_dx.shape[1], fh, fw)
        lh, rh = self.padding[0], pad_dx.shape[2] - self.padding[0]
        lw, rw = self.padding[1], pad_dx.shape[3] - self.padding[1]
        self.in_out['in'].set_error(pad_dx[:, :, lh: rh, lw: rw])
        return grads

    def convolution(self, x, flt, tmp_conv):
        # x [b, in_c, h+ph, w+pw]
        # flt [in_c, fh, fw, out_c]
        b = x.shape[0]
        fh, fw = flt.shape[1:3]
        sh, sw = self.stride

        for i in range(0, tmp_conv.shape[2]):
            for j in range(0, tmp_conv.shape[3]):
                seg_x = x[:, :, i*sh: i*sh+fh, j*sw: j*sw+fw].reshape(b, -1) # [b, in_c*h*w]
                _flt = flt.reshape(-1, flt.shape[-1]) # [in_c*fh*fw, out_c]
                tmp_conv[:, :, i, j] = seg_x @ _flt

        return tmp_conv


class Flatten(BaseLayer):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        self._x = self._process_input(x)
        out = self._x.reshape(self._x.shape[0], -1)
        warp_out = self._warp_out(out)
        return warp_out

    def backward(self):
        delta = self.in_out['out'].error
        grad = None
        self.in_out['in'].set_error(delta.reshape(self._x.shape))
        return grad


class Pool(BaseLayer):

    def __init__(self,
                 pool_size=(2, 2),
                 stride=2,
                 padding=0
                 ):
        super().__init__()

        self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)

        self.pad_img = None

    def forward(self, x):
        # x [b, in_c, h, w]
        self._x = self._process_input(x)

        self.pad_img, tmp_pool = get_pad_and_get_tmp(self._x, self.pool_size, self.stride, self._x.shape[1], self.padding)

        fh, fw = self.pool_size
        sh, sw = self.stride

        for i in range(tmp_pool.shape[2]):
            for j in range(tmp_pool.shape[3]):
                window = self.pad_img[:, :, i*sh: i*sh+fh, j*sw: j*sw+fw]
                tmp_pool[:, :, i, j] = self.special_pool(window)
        warp_out = self._warp_out(tmp_pool)
        return warp_out

    def special_pool(self, x):
        raise NotImplementedError


class Maxpool2D(Pool):

    def __init__(self,
                 pool_size=(2, 2),
                 stride=2,
                 padding=0
                 ):
        super().__init__(
            pool_size,
            stride,
            padding
        )

    def special_pool(self, x):
        return np.max(x, axis=(2, 3))

    def backward(self):
        delta = self.in_out['out'].error
        grads = None

        fh, fw = self.pool_size
        sh, sw = self.stride

        pad_dx = np.zeros_like(self.pad_img)
        for i in range(delta.shape[2]):
            for j in range(delta.shape[3]):
                window = self.pad_img[:, :, i*sh: i*sh+fh, j*sw: j*sw+fw]
                mask = window == np.max(window, axis=(2, 3), keepdims=True)
                window_delta = delta[:, :, i: i+1, j: j+1] * mask
                pad_dx[:, :, i*sh: i*sh+fh, j*sw: j*sw+fw] += window_delta

        lh, rh = self.padding[0], pad_dx.shape[2] - self.padding[0]
        lw, rw = self.padding[1], pad_dx.shape[3] - self.padding[1]
        self.in_out['in'].set_error(pad_dx[:, :, lh: rh, lw: rw])
        return grads


class Avgpool2D(Pool):

    def __init__(self,
                 pool_size=(2, 2),
                 stride=2,
                 padding=0
                 ):
        super().__init__(
            pool_size,
            stride,
            padding
        )

    def special_pool(self, x):
        return np.mean(x, axis=(2, 3))

    def backward(self):
        delta = self.in_out['out'].error
        grads = None

        fh, fw = self.pool_size
        sh, sw = self.stride

        pad_dx = np.zeros_like(self.pad_img)
        for i in range(delta.shape[2]):
            for j in range(delta.shape[3]):
                window_delta = delta[:, :, i: i+1, j: j+1] * 1.0 / (fh * fw)
                pad_dx[:, :, i*sh: i*sh+fh, j*sw: j*sw+fw] += window_delta

        lh, rh = self.padding[0], pad_dx.shape[2] - self.padding[0]
        lw, rw = self.padding[1], pad_dx.shape[3] - self.padding[1]
        self.in_out['in'].set_error(pad_dx[:, :, lh: rh, lw: rw])
        return grads


class Embedding(ParamLayer):

    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 w_init=None,
                 b_init=None,
                 ):
        super().__init__(
            w_shape=(num_embeddings, embedding_dim),
            w_init=w_init,
            b_init=b_init,
            activation=None,
            use_bias=None
        )

    def forward(self, ids):
        self._ids = self._process_input(ids)
        self._ids = self._ids.astype(np.int32)
        out = self.w[self._ids.astype(np.int32)]
        warp_out = self._warp_out(out)
        return warp_out

    def backward(self):
        delta = self.in_out['out'].error

        dw = np.zeros_like(self.w)
        for i in range(delta.shape[0]):
            dw[self._ids[i]] += delta[i]
        grads = {'w': dw}

        self.in_out['in'].set_error(None)
        return grads


def get_pad_and_get_tmp(img, kernel_size, stride, out_channels, padding):
    b, c, h, w = img.shape
    (fh, fw), (sh, sw) = kernel_size, stride
    ph, pw = padding

    pad_img = np.pad(img, ((0, 0), (0, 0), (ph, ph), (pw, pw)), 'constant', constant_values=0).astype(np.float32)

    h_out = int((h + 2 * ph - (fh - 1) - 1) / sh + 1)
    w_out = int((w + 2 * pw - (fw - 1) - 1) / sw + 1)
    tmp_data = np.zeros((b, out_channels, h_out, w_out), dtype=np.float32)
    return pad_img, tmp_data


if __name__ == '__main__':

    embedding = Embedding(3, 5)
    print(embedding.w)

    data = np.array([[1, 0, 0],
                     [0, 1, 0]])

    out = embedding.forward(data)
    print(out.data.shape)

    grads = embedding.backward()
    print(grads)