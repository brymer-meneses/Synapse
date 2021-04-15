
from .differentiable import Differentiable
from .gradFns import sumBackward, addBackward, mulBackward0, mulBackward1, \
                       powBackward, meanBackward, matmulBackward0, matmulBackward1, \
                      negBackward, subBackward0, subBackward1
from .types import Number

import numpy as np
from synapse import Tensor

@Differentiable(subBackward0, subBackward1)
def sub(t1: Tensor ,t2: Tensor) -> Tensor:
    data = t1.data + np.negative(t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    return Tensor(data, requires_grad)

@Differentiable(sumBackward)
def sum(t1: Tensor) -> Tensor:
    data = np.sum(t1.data)
    requires_grad = t1.requires_grad
    return Tensor(data, requires_grad)

@Differentiable(powBackward)
def pow(t1: Tensor, power: Number) -> Tensor:
    data = t1.data ** power
    requires_grad = t1.requires_grad
    return Tensor(data, requires_grad)

@Differentiable(meanBackward)
def mean(t1: Tensor) -> Tensor:
    data = np.mean(t1.data)
    requires_grad = t1.requires_grad
    return Tensor(data, requires_grad)

@Differentiable(negBackward)
def neg(t1: Tensor) -> Tensor:
    data = np.negative(t1.data)
    requires_grad =  t1.requires_grad
    return Tensor(data, requires_grad)

@Differentiable(addBackward)
def add(t1: Tensor, t2: Tensor) -> Tensor:
    if not t1.shape == t2.shape:
        raise ValueError("Broadcasting not Implemented (yet)")
    else:
        data = t1.data + t2.data

    requires_grad = t1.requires_grad or t2.requires_grad
    return Tensor(data, requires_grad)

@Differentiable(matmulBackward0, matmulBackward1)
def matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = np.matmul(t1.data, t2.data)

    requires_grad = t1.requires_grad or t2.requires_grad
    return Tensor(data, requires_grad)

@Differentiable(mulBackward0, mulBackward1)
def mul(t1: Tensor, t2: Tensor) -> Tensor:
    if not t1.shape == t2.shape:
        raise ValueError("Broadcasting not Implemented (yet)")
    else:
        data = t1.data * t2.data

    requires_grad = t1.requires_grad or t2.requires_grad
    return Tensor(data, requires_grad)

