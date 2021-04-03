
from typing import Callable
import numpy as np



def tensorSum(t1: 'Tensor') -> 'Tensor': #type: ignore

    from sparknet.autograd.tensor import Tensor, Node
    from sparknet.autograd.gradfns import sumBackward


    data = t1.data.sum()
    requiresGrad = t1.requiresGrad
    resultTensor: 'Tensor' = Tensor(data, requiresGrad)

    if requiresGrad:
        node = Node(t1, lambda grad: sumBackward(grad, t1))
        resultTensor.addParent(node)

    return resultTensor


def tensorAdd(t1: 'Tensor', t2: 'Tensor') -> 'Tensor': #type: ignore
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

def tensorMul(t1: 'Tensor', t2: 'Tensor') -> 'Tensor': #type: ignore

    from sparknet.autograd.tensor import Tensor, Node
    from sparknet.autograd.gradfns import mulBackward0, mulBackward1

    if t1.shape == t2.shape:
        data = t1.data * t2.data
    else:
        raise RuntimeError("Broadcasting not implemented")

    requiresGrad = t1.requiresGrad or t2.requiresGrad
    resultTensor = Tensor(data, requiresGrad)

    if t1.requiresGrad:
        node = Node(t1, lambda grad: mulBackward0(grad, t1, t2))
        resultTensor.addParent(node)
    if t2.requiresGrad:
        node = Node(t2, lambda grad: mulBackward1(grad, t1, t2))
        resultTensor.addParent(node)

    return resultTensor

def tensorMatMul(t1: 'Tensor', t2: 'Tensor') -> 'Tensor': #type: ignore
    from sparknet.autograd.tensor import Tensor, Node
    from sparknet.autograd.gradfns import matmulBackward1, matmulBackward0

    try:
        data = np.matmul(t1.data, t2.data)
    except Exception as e:
        raise RuntimeError(f"Caught Exception while trying to matrix-multiply tensors\n \
                            t1: {t1.shape}, t2: {t2.shape}")
    requiresGrad = t1.requiresGrad or t2.requiresGrad
    resultTensor = Tensor(data, requiresGrad)

    if t1.requiresGrad:
        node = Node(t1, lambda grad: matmulBackward0(grad, t1, t2))
        resultTensor.addParent(node)
    if t2.requiresGrad:
        node = Node(t2, lambda grad: matmulBackward1(grad, t1, t2))
        resultTensor.addParent(node)

    return resultTensor


