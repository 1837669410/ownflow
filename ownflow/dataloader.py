import numpy as np


class DataLoader:

    def __init__(self,
                 x,
                 y,
                 batch_size=32,
                 shuffle=True
                 ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ptr = 0

    def generate(self):
        while True:
            _p = self.ptr + self.batch_size
            # 超出数据范围则重新开始
            if _p > self.x.shape[0]:
                self.ptr = 0
                break

            # 初始化需要shuffle
            if self.ptr == 0 and self.shuffle:
                idx = np.random.permutation(self.x.shape[0])
                self.x[:] = self.x[idx]
                self.y[:] = self.y[idx]

            bx = self.x[self.ptr: _p]
            by = self.y[self.ptr: _p]
            self.ptr = _p
            yield bx, by

    def sample(self):
        return next(self.generate())