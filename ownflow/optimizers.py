import numpy as np


class Optimizer:

    def __init__(self, params, lr):
        super().__init__()

        self._params = params
        self._lr = lr
        self.vars = []
        self.grads = []
        for layer in self._params.keys():
            for key in self._params[layer]['params'].keys():
                self.vars.append(self._params[layer]['params'][key])
                self.grads.append(self._params[layer]['grads'][key])


class SGD(Optimizer):

    def __init__(self, params, lr=0.001, momentum=0.9, weight_decay=0):
        super().__init__(params, lr)

        self._momentum = momentum
        self._weight_decay = weight_decay
        self._mv = [np.zeros_like(v) for v in self.vars]

    def step(self):
        for var, grad, mv in zip(self.vars, self.grads, self._mv):
            dv = self._lr * (grad + self._weight_decay * var)
            mv[:] = self._momentum * mv + dv
            var -= mv


class Adagrad(Optimizer):

    def __init__(self, params, lr=0.01, lr_decay=0, weight_decay=0, eps=1e-8):
        super().__init__(params, lr)

        self._lr_decay = lr_decay
        self._weight_decay = weight_decay
        self._eps = eps
        self._mv = [np.zeros_like(v) for v in self.vars]
        self._init_t = 0

    def step(self):
        _lr = self._lr / (1 + self._init_t * self._lr_decay)
        for var, grad, mv in zip(self.vars, self.grads, self._mv):
            if self._weight_decay != 0:
                grad += self._weight_decay * var
            mv += grad ** 2
            var -= _lr * grad / (np.sqrt(mv) + self._eps)
        self._init_t += 1


class Adadelta(Optimizer):

    def __init__(self, params, lr=1.0, rho=0.9, weight_decay=0, eps=1e-6):
        super().__init__(params, lr)

        self._rho = rho
        self._weight_decay = weight_decay
        self._eps = eps
        self._v = [np.zeros_like(v) for v in self.vars]
        self._u = [np.zeros_like(v) for v in self.vars]

    def step(self):
        for var, grad, v, u in zip(self.vars, self.grads, self._v, self._u):
            if self._weight_decay != 0:
                grad += self._weight_decay * var
            v[:] = self._rho * v + (1 - self._rho) * grad ** 2
            delta_x = np.sqrt(u + self._eps) / np.sqrt(v + self._eps) * grad
            u[:] = self._rho * u + (1 - self._rho) * delta_x ** 2
            var -= self._lr * delta_x


class Adam(Optimizer):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), weight_decay=0, eps=1e-8):
        super().__init__(params, lr)

        self._beta1, self._beta2 = betas[0], betas[1]
        self._weight_decay = weight_decay
        self._eps = eps
        self._init_t = 0
        self._m = [np.zeros_like(v) for v in self.vars]
        self._v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        for var, grad, m, v in zip(self.vars, self.grads, self._m, self._v):
            if self._weight_decay != 0:
                grad += self._weight_decay * var
            m[:] = self._beta1 * m + (1 - self._beta1) * grad
            v[:] = self._beta2 * v + (1 - self._beta2) * grad ** 2
            mt = m / (1 - self._beta1 ** (self._init_t + 1))
            vt = v / (1 - self._beta2 ** (self._init_t + 1))
            var -= self._lr * mt / (np.sqrt(vt) + self._eps)
        self._init_t += 1


class AdamW(Optimizer):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        super().__init__(params, lr)

        self._beta1, self._beta2 = betas[0], betas[1]
        self._weight_decay = weight_decay
        self._eps = eps
        self._init_t = 0
        self._m = [np.zeros_like(v) for v in self.vars]
        self._v = [np.zeros_like(v) for v in self.vars]

    def step(self):
        for var, grad, m, v in zip(self.vars, self.grads, self._m, self._v):
            var -= self._lr * self._weight_decay * var
            m[:] = self._beta1 * m + (1 - self._beta1) * grad
            v[:] = self._beta2 * v + (1 - self._beta2) * grad ** 2
            mt = m / (1 - self._beta1 ** (self._init_t + 1))
            vt = v / (1 - self._beta2 ** (self._init_t + 1))
            var -= self._lr * mt / (np.sqrt(vt) + self._eps)
        self._init_t += 1


class Adamax(Optimizer):

    def __init__(self, params, lr=0.002, betas=(0.9, 0.999), weight_decay=0, eps=1e-8):
        super().__init__(params, lr)

        self._beta1, self._beta2 = betas[0], betas[1]
        self._weight_decay = weight_decay
        self._eps = eps
        self._m = [np.zeros_like(v) for v in self.vars]
        self._u = [np.zeros_like(v) for v in self.vars]
        self._init_t = 0

    def step(self):
        for var, grad, m, u in zip(self.vars, self.grads, self._m, self._u):
            if self._weight_decay != 0:
                grad += self._weight_decay * var
            m[:] = self._beta1 * m + (1 - self._beta1) * grad
            u[:] = np.maximum(self._beta2 * u, np.abs(grad) + self._eps)
            var -= self._lr * m / (1 - self._beta1 ** (self._init_t + 1)) / u
        self._init_t += 1


if __name__ == '__main__':

    import ownflow as of

    class Net(of.module.Module):

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
    opt = SGD(model.params, 0.1)

    # print(model.params['linear1'])
    # print(model.params['linear2'])

    out2 = model.forward(x)
    loss = out2.data - y

    model.backward(loss)

    print(model.params['linear1']['params'])

    opt.step()

    print(model.params['linear1']['params'])