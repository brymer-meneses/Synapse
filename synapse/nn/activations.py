
from synapse.nn.layers import Layer
from synapse.core.tensor import Tensor

from synapse.core.differentiable import Differentiable

import numpy as np

from typing import Callable

def tanhBackward(grad: Tensor, t1: Tensor) -> Tensor:
    data = grad.data * (1 - np.tanh(t1.data) ** 2)
    return Tensor(data)

@Differentiable(tanhBackward)
def Tanh(t1: Tensor) -> Tensor:
    data = np.tanh(t1.data)
    requires_grad = t1.requires_grad

    return Tensor(data, requires_grad)

def reluBackward(grad: Tensor, t1: Tensor) -> Tensor:
    data = grad.data * np.where(t1.data > 0, 1, 0)
    return Tensor(data)

@Differentiable(reluBackward)
def ReLU(t1: Tensor) -> Tensor:
    data = np.maximum(0, t1.data, t1.data) # Use in place operation
    return Tensor(data, t1.requires_grad)


class Softmax():
    def forward(self, t1: Tensor) -> Tensor:
        expData = np.exp(t1.data)
        data = expData / np.sum(expData, axis=0)
        requires_grad = t1.requires_grad
        return Tensor(expData, requires_grad)

    def gradFn(self, t1: Tensor) -> Callable[[np.ndarray], Tensor]:
        def SoftmaxBackward(grad: np.ndarray) -> Tensor:
            """TODO"""
            pass

            return
        return







