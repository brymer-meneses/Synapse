from synapse.core.tensor import Node, Tensor
from synapse.core.differentiable import Differentiable
import synapse as sn
from abc import abstractmethod

import numpy as np
from typing import Callable

def MSE(predicted: Tensor, target: Tensor) -> Tensor:
    return sn.pow(target - predicted, 2).mean()



class CategoricalCrossEntropy():
    """TODO"""
    def forward(self, x, y) -> Tensor:
        if np.any(x < 0) or np.any(y < 0):
            raise ValueError("Only non-zero values are allowed as an input to this function")

        x = np.array(x, dtype=np.float)
        y = np.array(y, dtype=np.float)
        x /= np.sum(x)
        y /= np.sum(y)

        mask = y > 0
        y = y[mask]
        x = x[mask]

        result = -np.sum(x*np.log(y))

        return Tensor(result, x.requiresGrad)






