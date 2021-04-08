
from ._differentiable import Differentiable
from ._gradFns import sumBackward, addBackward, mulBackward0, mulBackward1, \
                       powBackward, meanBackward, matmulBackward0, matmulBackward1, \
                      negBackward
from ._types import Number

import numpy as np
from synapse import Tensor


def sum(t1: Tensor) -> Tensor:
    data = np.sum(t1.data)
    requiresGrad = t1.requiresGrad
    return Tensor(data, requiresGrad)

def pow(t1: Tensor, power: Number) -> Tensor:
    data = t1.data ** power
    requiresGrad = t1.requiresGrad
    return Tensor(data, requiresGrad)

def mean(t1: Tensor) -> Tensor:
    data = np.mean(t1.data)
    requiresGrad = t1.requiresGrad
    return Tensor(data, requiresGrad)

def neg(t1: Tensor) -> Tensor:
    data = np.negative(t1.data)
    requiresGrad =  t1.requiresGrad
    return Tensor(data, requiresGrad)

def add(t1: Tensor, t2: Tensor) -> Tensor:
    if not t1.shape == t2.shape:
        raise ValueError("Broadcasting not Implemented (yet)")
    else:
        data = t1.data + t2.data

    requiresGrad = t1.requiresGrad
    return Tensor(data, requiresGrad)

def matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = np.matmul(t1.data, t2.data)

    requiresGrad = t1.requiresGrad or t2.requiresGrad
    return Tensor(data, requiresGrad)

def mul(t1: Tensor, t2: Tensor) -> Tensor:
    if not t1.shape == t2.shape:
        raise ValueError("Broadcasting not Implemented (yet)")
    else:
        data = t1.data * t2.data

    requiresGrad = t1.requiresGrad or t2.requiresGrad
    return Tensor(data, requiresGrad)

add = Differentiable(add, addBackward)
sum = Differentiable(sum, sumBackward)
matmul = Differentiable(matmul, matmulBackward0, matmulBackward1)
mul = Differentiable(mul, mulBackward0, mulBackward1)
pow = Differentiable(pow, powBackward)
mean = Differentiable(mean, meanBackward)
neg = Differentiable(neg, negBackward)

