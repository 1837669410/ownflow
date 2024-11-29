from abc import abstractmethod

import numpy as np

import ownflow as of


class Module:

    def __init__(self):
        self.params = {}
        self.order_layers = []

    @abstractmethod
    def forward(self, x):
        pass

    def backward(self, loss):
        layers = []
        for key, value in self.__dict__.items():
            if not isinstance(value, of.layers.BaseLayer):
                continue
            layer = value
            layer.name = key
            layers.append((layer.order, layer))

        self.order_layers = [l[1] for l in sorted(layers, key=lambda x: x[0])]
        self.order_layers[-1].in_out['out'].set_error(loss)

        for layer in self.order_layers[::-1]:
            grads = layer.backward()
            for k, v in grads.items():
                self.params[layer.name]['grads'][k][:] = v

    def __setattr__(self, key, value):
        if isinstance(value, of.layers.ParamLayer):
            layer = value
            self.params[key] = {
                'params': layer.params,
                'grads': {k: np.empty_like(v, dtype=np.float32) for k, v in layer.params.items()}
            }
        super().__setattr__(key, value)


if __name__ == '__main__':
    class Net(Module):

        def __init__(self):
            super().__init__()

            import ownflow as of

            self.linear1 = of.layers.Linear(3, 4)
            self.linear2 = of.layers.Linear(4, 1)

        def forward(self, x):
            out1 = self.linear1.forward(x)
            out2 = self.linear2.forward(out1)
            return out2

    x = np.ones((2, 3))
    y = np.ones((2, 1))

    model = Net()

    print(model.params['linear1']['grads'])
    print(model.params['linear2']['grads'])

    out2 = model.forward(x)
    loss = out2.data - y

    model.backward(loss)

    for layer in model.params.keys():
        print(model.params[layer].keys())