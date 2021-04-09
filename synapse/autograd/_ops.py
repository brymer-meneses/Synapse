
from ._differentiable import Differentiable
from ._gradFns import sumBackward, addBackward, mulBackward0, mulBackward1, \
                       powBackward, meanBackward, matmulBackward0, matmulBackward1, \
                      negBackward, subBackward
from ._types import Number

import numpy as np
from synapse import Tensor

@Differentiable(subBackward)
def sub(t1: Tensor ,t2: Tensor) -> Tensor:
    data = t1.data + np.negative(t2.data)
    requiresGrad = t1.requiresGrad or t2.requiresGrad
    return Tensor(data, requiresGrad)

@Differentiable(sumBackward)
def sum(t1: Tensor) -> Tensor:
    data = np.sum(t1.data)
    requiresGrad = t1.requiresGrad
    return Tensor(data, requiresGrad)

@Differentiable(powBackward)
def pow(t1: Tensor, power: Number) -> Tensor:
    data = t1.data ** power
    requiresGrad = t1.requiresGrad
    return Tensor(data, requiresGrad)

@Differentiable(meanBackward)
def mean(t1: Tensor) -> Tensor:
    data = np.mean(t1.data)
    requiresGrad = t1.requiresGrad
    return Tensor(data, requiresGrad)

@Differentiable(negBackward)
def neg(t1: Tensor) -> Tensor:
    data = np.negative(t1.data)
    requiresGrad =  t1.requiresGrad
    return Tensor(data, requiresGrad)

@Differentiable(addBackward)
def add(t1: Tensor, t2: Tensor) -> Tensor:
    if not t1.shape == t2.shape:
        raise ValueError("Broadcasting not Implemented (yet)")
    else:
        data = t1.data + t2.data

    requiresGrad = t1.requiresGrad or t2.requiresGrad
    return Tensor(data, requiresGrad)

@Differentiable(matmulBackward0, matmulBackward1)
def matmul(t1: Tensor, t2: Tensor) -> Tensor:
    data = np.matmul(t1.data, t2.data)

    requiresGrad = t1.requiresGrad or t2.requiresGrad
    return Tensor(data, requiresGrad)

@Differentiable(mulBackward0, mulBackward1)
def mul(t1: Tensor, t2: Tensor) -> Tensor:
    if not t1.shape == t2.shape:
        raise ValueError("Broadcasting not Implemented (yet)")
    else:
        data = t1.data * t2.data

    requiresGrad = t1.requiresGrad or t2.requiresGrad
    return Tensor(data, requiresGrad)

