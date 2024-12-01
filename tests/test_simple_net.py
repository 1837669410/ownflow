import math

import numpy as np
import matplotlib.pyplot as plt

import ownflow as of


class Net(of.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = of.layers.Linear(1, 10,
                                        w_init=of.initializers.Kaiming_Uniform_(a=math.sqrt(5)),
                                        activation=of.activations.Relu())
        self.linear2 = of.layers.Linear(10, 10,
                                        w_init=of.initializers.Kaiming_Uniform_(a=math.sqrt(5)),
                                        activation=of.activations.Relu())
        self.linear3 = of.layers.Linear(10, 1,
                                        w_init=of.initializers.Kaiming_Uniform_(a=math.sqrt(5)),)

    def forward(self, x):
        out = self.linear1.forward(x)
        out = self.linear2.forward(out)
        out = self.linear3.forward(out)
        return out


if __name__ == '__main__':

    x = np.linspace(-2, 2, 100)[:, None]
    y = x ** 3 + np.random.normal(0., 0.1, x.shape)

    model = Net()
    print(model.params['linear1']['params'])
    print(model.params['linear2']['params'])
    print(model.params['linear3']['params'])
    opt = of.optimizers.SGD(model.params, lr=0.001)

    plt.ion()

    for i in range(200):
        out = model.forward(x)

        cost = np.sum((out.data - y) ** 2) / 2

        loss = out.data - y

        model.backward(loss)

        opt.step()

        print(cost)

        plt.plot(x, y, label='true')
        plt.plot(x, model.forward(x).data, label='pred')
        plt.legend()
        plt.title(f'epoch:{i}')
        plt.pause(0.1)
        plt.cla()




