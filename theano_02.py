#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

x = T.dmatrix('x')

s = 1 / (1 + T.exp(-x))

sigmoid = theano.function(inputs=[x], outputs=s)

print sigmoid([[0, 1], [-1, -2]])
