import math

import numpy as np
import matplotlib.pyplot as plt

import ownflow as of


class Word2Vec(of.Module):

    def __init__(self, n_vocab, out_dim):
        super().__init__()

        self.embedding = of.layers.Embedding(n_vocab, out_dim,
                                             w_init=of.initializers.Normal_(0, 0.1))

    def forward(self, x):
        out = self.embedding.forward(x)
        out = np.mean(out.data, axis=1)
        # out = self.linear1.forward(out)
        return out


if __name__ == '__main__':

    text = [
        # numbers
        "5 2 4 8 6 2 3 6 4",
        "4 8 5 6 9 5 5 6",
        "1 1 5 2 3 3 8",
        "3 6 9 6 8 7 4 6 3",
        "8 9 9 6 1 4 3 4",
        "1 0 2 0 2 1 3 3 3 3 3",
        "9 3 3 0 1 4 7 8",
        "9 9 8 5 6 7 1 2 3 0 1 0",

        # alphabets, expecting that 9 is close to letters
        "a t g q e h 9 u f",
        "e q y u o i p s",
        "q o 9 p l k j o k k o p",
        "h g y i u t t a e q",
        "i k d q r e 9 e a d",
        "o p d g 9 s a f g a",
        "i u y g h k l a s w",
        "o l u y a o g f s",
        "o 9 i u y g d a s j d l",
        "u k i l o 9 l j s",
        "y g i s h k j l f r f",
        "i o h n 9 9 d 9 f a 9",
    ]

    vocab = sum([line.split() for line in text], [])
    vocab = set(vocab)

    i2v = {i: v for i, v in enumerate(vocab)}
    v2i = {v: i for i, v in enumerate(vocab)}

    # word2vec数据切分
    x = []
    y = []
    for line in text:
        tmp = line.split()
        for i in range(2, len(tmp)-2):
            x.append([v2i[tmp[i-2]], v2i[tmp[i-1]], v2i[tmp[i+1]], v2i[tmp[i+2]]])
            y.append(v2i[tmp[i]])
    x = np.array(x)
    y = np.array(y)

    dataloader = of.dataloader.DataLoader(x, y, batch_size=4, shuffle=True)

    model = Word2Vec(len(vocab), len(vocab))
    loss_func = of.losses.CrossEntropyLoss()
    opt = of.optimizers.Adagrad(model.params, lr=0.01, lr_decay=1e-5)

    for i in range(3000):
        for step, (x, y) in enumerate(dataloader.generate()):

            out = model.forward(x)

            loss, loss_delta = loss_func.forward(out, y)

            model.backward(loss_delta)

            opt.step()

        if i % 500 == 0:
            print(f'iter:{i} | loss:{loss:.4f}')

    w = model.embedding.w

    w = (w - np.mean(w, axis=0)) / np.std(w, axis=0)
    cov = np.cov(w)
    eig, eigvec = np.linalg.eig(cov)

    idx = np.argsort(eig)[::-1]
    eigvec = eigvec[:, idx][:, :2]

    for i in range(len(vocab)):
        c = "blue"
        try:
            int(i2v[i])
        except ValueError:
            c = "red"
        plt.text(eigvec[i, 0], eigvec[i, 1], s=i2v[i], color=c, weight="bold")
    plt.xlim(eigvec[:, 0].min() - .5, eigvec[:, 0].max() + .5)
    plt.ylim(eigvec[:, 1].min() - .5, eigvec[:, 1].max() + .5)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel("embedding dim1")
    plt.ylabel("embedding dim2")
    plt.show()