
import numpy as np
from tensor import Tensor, Node

from gradfns import sumBackward


def tensorSum(tensor: Tensor) -> Tensor:

    data = tensor.data.sum()
    requiresGrad = tensor.requiresGrad
    resultTensor: Tensor = Tensor(data, requiresGrad)

    if requiresGrad:
        node = Node(tensor, sumBackward)
        resultTensor.__addParent(node)

    return resultTensor


def tensorAdd(t1: Tensor, t2: Tensor) -> Tensor:

    data = t1.data + t2.data
    requiresGrad = t1.requiresGrad or t2.requiresGrad

    # ToDo
