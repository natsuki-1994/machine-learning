#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F

plt.style.use('ggplot')

batchsize = 100  # 確率的勾配降下法で学習させる際の1回分のバッチサイズ
n_epoch = 20  # 学習の繰り返し回数
n_units = 1000  # 隠れ層の数

# minstデータのダウンロード
print('downloading mnist data...')
mnist = fetch_mldata('MNIST original')
print('finish!')
# mnist.data : 70,000件の784次元ベクトルデータ
mnist.data = mnist.data.astype(np.float32)
mnist.data /= 255  # 0-1データに変換

# mnist.target : 正解データ
mnist.target = mnist.target.astype(np.int32)


# 手書き数字を描画する関数
def draw_digit(data):
    size = 28
    plt.figure(figsize=(2.5, 3))

    X, Y = np.meshgrid(range(size), range(size))
    Z = data.reshape(size, size)  # 784次元ベクトルデータを28*28配列に変換
    Z = Z[::-1, :]  # 上下反転
    plt.xlim(0, 27)
    plt.ylim(0, 27)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft="off")

    plt.show()

'''
# 手書き数字を描画
draw_digit(mnist.data[5])
draw_digit(mnist.data[12345])
draw_digit(mnist.data[33456])
'''

# 学習用データをN個, 検証用データを残りの個数に設定
N = 60000
x_train, x_test = np.split(mnist.data, [N])
y_train, y_test = np.split(mnist.target, [N])
N_test = y_test.size

# 多層パーセプトロンモデルの設定
# 入力784次元, 出力10次元
# 隠れ層は1000次元
model = FunctionSet(l1=F.Linear(784, n_units),
                    l2=F.Linear(n_units, n_units),
                    l3=F.Linear(n_units, 10))


# ニューラルネットの順伝播の構造を定義
def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)  # データは配列からChainer特有のVariable型（クラス）のオブジェクトに変換
    h1_ = F.dropout(F.relu(model.l1(x)), train=train)
    h2_ = F.dropout(F.relu(model.l2(h1_)), train=train)
    y_ = model.l3(h2_)

    # 誤差関数としてソフトマックス関数の交差エントロピー関数を用いて誤差を導出
    return F.softmax_cross_entropy(y_, t), F.accuracy(y_, t)

# 最適化手法（optimizer）の設定
# optimizerはパラメータと勾配からなり、update()を実行するたびに対応する勾配に基づきパラメータを更新
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

train_loss = []
train_acc = []
test_loss = []
test_acc = []

l1_W = []
l2_W = []
l3_W = []


# 学習
for epoch in xrange(1, n_epoch+1):
    print('epoch: {}'.format(epoch))

    perm = np.random.permutation(N)  # N個の順番をランダムに並び替える
    sum_accuracy = 0
    sum_loss = 0

    # 0-Nまでのデータをバッチサイズごとに使って学習
    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]

        # 勾配を初期化
        optimizer.zero_grads()
        # 順伝播させて誤差と精度を算出
        loss, acc = forward(x_batch, y_batch)
        # 誤差逆伝播で勾配を計算
        loss.backward()
        optimizer.update()

        train_loss.append(loss.data)
        train_acc.append(acc.data)
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    # 訓練データの誤差と、正解精度を表示
    print('train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N))

    # テストデータでの誤差と正解精度を算出し汎化性能を確認
    sum_accuracy = 0
    sum_loss = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]

        # 順伝播させて誤差と精度を算出
        loss, acc = forward(x_batch, y_batch)

        test_loss.append(loss.data)
        test_acc.append(acc.data)
        sum_loss += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    # テストデータの誤差と、正解精度を表示
    print('test mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test))

    # 学習したパラメータを保存
    l1_W.append(model.l1.W)
    l2_W.append(model.l2.W)
    l3_W.append(model.l3.W)

# 精度と誤差をグラフ描画
plt.figure(figsize=(8, 6))
plt.plot(range(len(train_acc)), train_acc)
plt.plot(range(len(test_acc)), test_acc)
plt.legend(['train_acc', 'test_acc'], loc=4)
plt.title('Accuracy of digit recognition')
plt.savefig('plt-mnist-01.png')

plt.style.use('fivethirtyeight')


# 学習したモデルを使用して答え合わせ
def draw_digit3(data, n, ans, recog):
    size = 28
    plt.subplot(10, 10, n)
    Z = data.reshape(size, size)
    Z = Z[::-1, :]  # 上下反転
    plt.xlim(0, 27)
    plt.ylim(0, 27)
    plt.pcolor(Z)
    plt.title('ans={}, recog={}'.format(ans, recog), size=8)
    plt.gray()
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')

plt.figure(figsize=(15, 15))
cnt = 0
for idx in np.random.permutation(N)[:100]:
    xxx = x_train[idx].astype(np.float32)
    h1 = F.dropout(F.relu(model.l1(Variable(xxx.reshape(1, 784)))), train=False)
    h2 = F.dropout(F.relu(model.l2(h1)), train=False)
    y = model.l3(h2)
    cnt += 1
    draw_digit3(x_train[idx], cnt, y_train[idx], np.argmax(y.data))

plt.show()


def draw_digit2(data, n, i_):
    size = 28
    plt.subplot(10, 10, n)
    Z = data.reshape(size, size)
    Z = Z[::-1, :]  # 上下反転
    plt.xlim(0, 27)
    plt.ylim(0, 27)
    plt.pcolor(Z)
    plt.title('{}'.format(i_), size=9)
    plt.gray()
    plt.tick_params(labelbottom='off')
    plt.tick_params(labelleft='off')

plt.figure(figsize=(10, 10))
cnt = 1
for i in np.random.permutation(1000)[:100]:
    draw_digit2(l1_W[len(l1_W) - 1][i], cnt, i)  # len(l1_W) : epoch数
    cnt += 1

plt.show()
