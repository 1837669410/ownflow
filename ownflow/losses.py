import numpy as np


class Loss:

    def __init__(self):
        super().__init__()

        self._eps = 1e-6

    def forward(self, pred, target):
        pass

    def __call__(self, pred, target):
        return self.forward(pred, target)


def _one_hot(x, depth):
    x_one_hot = np.zeros((x.shape[0], depth))
    x_one_hot[np.arange(x.shape[0]), x] = 1
    return x_one_hot


class CrossEntropyLoss(Loss):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        target = target if target.dtype == np.int32 else target.astype(np.int32)
        target = _one_hot(target, pred.shape[-1])
        pred = pred.data

        pred -= np.max(pred, axis=-1, keepdims=True) # 稳定softmax
        pred = np.exp(pred + self._eps) / np.sum(np.exp(pred + self._eps), axis=-1, keepdims=True)

        loss = -np.mean(target * np.log(pred + self._eps))

        loss_delta = (pred - target) / pred.shape[0]

        return loss, loss_delta
