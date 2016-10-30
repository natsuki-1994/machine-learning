#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

x = T.dscalar('x')
print type(x)

y = x ** 2
print type(y)

# 関数を定義
# コンパイルはここで実行される
f = theano.function(inputs=[x], outputs=y)
print type(f)

print f(1)
print f(2)

