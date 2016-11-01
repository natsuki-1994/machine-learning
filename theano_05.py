#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

"""
2クラスロジスティック回帰をtheanoで実装
"""


def plot_data(X, y):
    # positiveクラスのデータのインデックス
    positive = [i for i in range(len(y)) if y[i] == 1]
    # negativeクラスのデータのインデックス
    negative = [i for i in range(len(y)) if y[i] == 0]

    plt.scatter(X[positive, 0], X[positive, 1], c='red', marker='o', label='positive')
    plt.scatter(X[negative, 0], X[negative, 1], c='blue', marker='o', label='negative')

if __name__ == '__main__':
    # 訓練データのロード
    data = np.genfromtxt("ex2data1.txt", delimiter=',')
    data_x = data[:, (0, 1)]
    data_y = data[:, 2]

    # 訓練データ数
    m = len(data_y)

    # 訓練データのプロット
    plt.figure(1)
    plot_data(data_x, data_y)