import math

import numpy as np

import ownflow as of


class SimpleNet(of.Module):

    def __init__(self):
        super().__init__()

        self.linear1 = of.layers.Linear(784, 256,
                                        w_init=of.initializers.Kaiming_Uniform_(math.sqrt(5)),
                                        activation=of.activations.Relu())
        self.linear2 = of.layers.Linear(256, 100,
                                        w_init=of.initializers.Kaiming_Uniform_(math.sqrt(5)),
                                        activation=of.activations.Relu())
        self.linear3 = of.layers.Linear(100, 10,
                                        w_init=of.initializers.Kaiming_Uniform_(math.sqrt(5)))

    def forward(self, x):
        x = x.reshape(-1, 784)
        out = self.linear1.forward(x)
        out = self.linear2.forward(out)
        out = self.linear3.forward(out)
        return out


if __name__ == '__main__':
    data = np.load('./mnist.npz')
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

    train_loader = of.dataloader.DataLoader(x_train, y_train, batch_size=64)
    test_loader = of.dataloader.DataLoader(x_test, y_test, batch_size=64)

    model = SimpleNet()
    loss_func = of.losses.CrossEntropyLoss()
    opt = of.optimizers.SGD(model.params, 0.001)

    for i in range(20):
        for step, (x, y) in enumerate(train_loader.generate()):
            x = x / 255. * 2. - 1.

            out = model.forward(x)

            loss, loss_delta = loss_func.forward(out, y)

            model.backward(loss_delta)

            opt.step()

            if step % 200 == 0:
                print(f'iter:{i} | step:{step} | loss:{loss:.4f}')

        total_num = 0
        total_acc = 0
        for step, (x, y) in enumerate(test_loader.generate()):
            x = x / 255. * 2. - 1.
            out = model.forward(x)

            total_acc += np.sum(np.argmax(out.data, axis=1) == y)
            total_num += x.shape[0]
        print(f'iter:{i} | acc:{total_acc / total_num:.4f}')