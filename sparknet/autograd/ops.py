
from typing import Callable
import numpy as np



def tensorSum(tensor: 'Tensor') -> 'Tensor':

    from sparknet.autograd.tensor import Tensor, Node
    from sparknet.autograd.gradfns import sumBackward


    data = tensor.data.sum()
    requiresGrad = tensor.requiresGrad
    resultTensor: Tensor = Tensor(data, requiresGrad)

    if requiresGrad:
        node = Node(tensor, lambda grad: sumBackward(grad, t1, t2))
        resultTensor.addParent(node)

    return resultTensor


def tensorAdd(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    """Performs element wise addition to two Tensors"""

    from sparknet.autograd.tensor import Tensor, Node
    from sparknet.autograd.gradfns import addBackward

    if t1.shape == t2.shape:
        data = t1.data + t2.data
    else:
        raise RuntimeError("Broadcasting not implemented")
        pass

    requiresGrad = t1.requiresGrad or t2.requiresGrad
    resultTensor = Tensor(data, requiresGrad)

    if t1.requiresGrad:
        node = Node(t1, lambda grad: addBackward(grad, t1, t2))
        resultTensor.addParent(node)
    if t2.requiresGrad:
        node = Node(t2, lambda grad: addBackward(grad, t1, t2))
        resultTensor.addParent(node)

    return resultTensor

def tensorMul(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':

    from sparknet.autograd.tensor import Tensor, Node
    from sparknet.autograd.gradfns import mulBackward

    if t1.shape == t2.shape:
        data = t1.data * t2.data
    else:
        raise RuntimeError("Broadcasting not implemented")
        pass

    requiresGrad = t1.requiresGrad or t2.requiresGrad
    resultTensor = Tensor(data, requiresGrad)

    if t1.requiresGrad:
        node = Node(t1, lambda grad: mulBackward(grad, t1, t2))
        resultTensor.addParent(node)
    if t2.requiresGrad:
        node = Node(t2, lambda grad: mulBackward(grad, t1, t2))
        resultTensor.addParent(node)

    return resultTensor

def tensorMatMul(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    from sparknet.autograd.tensor import Tensor, Node
    from sparknet.autograd.gradfns import matmulBackward1, matmulBackward0

    data = np.matmul(t1.data, t2.data)
    requiresGrad = t1.requiresGrad or t2.requiresGrad
    resultTensor = Tensor(data, requiresGrad)

    if t1.requiresGrad:
        node = Node(t1, lambda grad: matmulBackward0(grad, t1, t2))
        resultTensor.addParent(node)
    if t2.requiresGrad:
        node = Node(t2, lambda grad: matmulBackward1(grad, t1, t2))
        resultTensor.addParent(node)

    return resultTensor


