import math

import numpy as np

import ownflow as of


class SimpleCNN(of.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = of.layers.Conv2D(1, 8, (3, 3), 2, 1,
                                      w_init=of.initializers.Kaiming_Uniform_(math.sqrt(5)),
                                      activation=of.activations.Gelu())
        self.conv2 = of.layers.Conv2D(8, 16, (3, 3), 2, 1,
                                      w_init=of.initializers.Kaiming_Uniform_(math.sqrt(5)),
                                      activation=of.activations.Gelu())
        self.conv3 = of.layers.Conv2D(16, 32, (3, 3), 2, 1,
                                      w_init=of.initializers.Kaiming_Uniform_(math.sqrt(5)),
                                      activation=of.activations.Gelu())
        self.flatten = of.layers.Flatten()
        self.linear1 = of.layers.Linear(32 * 4 * 4, 10,
                                        w_init=of.initializers.Kaiming_Uniform_(math.sqrt(5)))

    def forward(self, x):
        b, h, w = x.shape
        x = x.reshape(b, 1, h, w)
        # [b, 1, 28, 28] -> [b, 8, 14, 14]
        out = self.conv1.forward(x)
        # [b, 8, 14, 14] -> [b, 16, 7, 7]
        out = self.conv2.forward(out)
        # [b, 16, 7, 7] -> [b, 32, 4, 4]
        out = self.conv3.forward(out)
        # [b, 32, 4, 4] -> [b, 32 * 4 * 4]
        out = self.flatten.forward(out)
        # [b, 32 * 4 * 4] -> [b, 10]
        out = self.linear1.forward(out)
        return out


if __name__ == '__main__':
    data = np.load('./mnist.npz')
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

    train_loader = of.dataloader.DataLoader(x_train, y_train, batch_size=64)
    test_loader = of.dataloader.DataLoader(x_test, y_test, batch_size=64)

    model = SimpleCNN()
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