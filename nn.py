#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as datasets
import sklearn.linear_model as linear_model
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

# データを生成してプロット
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)  # X...2次元座標[i, j], y...0 or 1
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)  # x座標, y座標, ...
plt.savefig('plt-01.png')

# ロジスティクス回帰モデルを学習させる
clf = linear_model.LogisticRegressionCV()
clf.fit(X, y)


# predictして境界線を引く関数
def plot_decision_boundary(pred_func):

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # グリッド生成
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # pred_func実行
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # プロット
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)

# plot_decision_boundary(lambda x: clf.predict(x))
# plt.title("Logistic Regression")
# plt.savefig('plt-02.png')

num_examples = len(X)  # 学習用データサイズ
nn_input_dim = 2  # インプット用次元数（縦×横）
nn_output_dim = 2  # アウトプット用次元数（青, 赤）

# 勾配降下法パラメータ
epsilon = 0.01  # 学習率
reg_lambda = 0.01  # 正則化の強さ


# 全lossを計算するためのHelper関数
def calculate_loss(model):

    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # 予測を算出するための順方向伝搬
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # lossを計算
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)

    # lossに正則化項を追加
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss


# アウトプット（0 or 1）を推測するためのHelper関数
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # 順方向伝搬
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    print np.argmax(probs, axis=1)


# ニューラルネットワークのパラメータを学習しモデルを返す関数
# - nn_hdim : 隠れ層の次元数
# - num_passes: 繰り返しの数
# - print_loss: Trueならlossをprint
def build_model(nn_hdim, num_passes=20000, print_loss=False):

    # パラメータの初期化
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}

    # バッチごとに勾配法
    for i in xrange(0, num_passes):

        # 順方向伝搬
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 逆方向伝搬
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = a1.T.dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # 正則化項の追加
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # パラメータの更新
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # lossをprint, ただし時間がかかる
        if print_loss and i % 1000 == 0:
            print("{} {}".format(i, calculate_loss(model)))

    return model

# 3次元の隠れ層を持つモデルの構築
best_model = build_model(3, print_loss=True)

# 決定境界をプロットする
plot_decision_boundary(lambda x: predict(best_model, x))
plt.title("Decision Boundary for hidden layer size 3")

plt.savefig('plt-03.png')
