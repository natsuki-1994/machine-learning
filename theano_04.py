#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T

x = T.dscalar('x')

y = x ** 2

# yをxに関して微分
gy = T.grad(cost=y, wrt=x)
print type(gy)

f = theano.function(inputs=[x], outputs=gy)
print theano.pp(f.maker.fgraph.outputs[0])

print f(2)
print f(3)
