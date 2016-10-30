#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.datasets
import matplotlib
import matplotlib.pyplot as plt

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
plt.savefig('plt-chainer-01.png')


def plot_decision_boundary(pred_func):

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

n_units = 3  # 隠れ層の数


# Chain作成
class Model(Chain):
    def __init__(self):
        super(Model, self).__init__(
            l1=L.Linear(2, n_units),
            l2=L.Linear(n_units, 2),
        )

    def __call__(self, x_):
        h1 = F.tanh(self.l1(x_))
        y_ = self.l2(h1)
        return y_

# Classifier Chain作成
model_ = L.Classifier(Model())

# optimizer作成
optimizer = optimizers.Adam()
optimizer.setup(model_)

# 学習
x = Variable(X.astype(np.float32))
t = Variable(y.astype(np.int32))

for _ in range(20000):
    optimizer.update(model_, x, t)


def predict(model, x_data):
    x_ = Variable(x_data.astype(np.float32))
    y_ = model.predictor(x_)
    return np.argmax(y_.data, axis=1)

plot_decision_boundary(lambda x_: predict(model_, x_))
plt.savefig('plt-chainer-02.png')
