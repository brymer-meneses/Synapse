
import numpy as np



def tensorSum(tensor: 'Tensor') -> 'Tensor':
    from sparknet.autograd.tensor import Tensor

    data = tensor.data.sum()
    requiresGrad = tensor.requiresGrad
    resultTensor: Tensor = Tensor(data, requiresGrad)

    if requiresGrad:
        node = Node(tensor, sumBackward)
        resultTensor.__addParent(node)

    return resultTensor


def tensorAdd(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    """Performs element wise addition to two Tensors"""
    from sparknet.autograd.tensor import Tensor

    if t1.shape == t2.shape:
        data = t1.data + t2.data
    else:
        raise RuntimeError("Broadcasting not implemented")
        pass

    requiresGrad = t1.requiresGrad or t2.requiresGrad
    resultTensor = Tensor(data, requiresGrad)

    if t1.requiresGrad:
        resultTensor.__addParent(t1, addBackward)
    if t2.requiresGrad:
        resultTensor.__addParent(t2, addBackward)

    return resultTensor

def tensorMatMul(t1: 'Tensor', t2: 'Tensor') -> 'Tensor':
    from sparknet.autograd.tensor import Tensor

    data = np.matmul(t1.data, t2.data)
    requiresGrad = t1.requiresGrad or t2.requiresGrad
    resultTensor = Tensor(data, requiresGrad)

    if t1.requiresGrad:
        resultTensor.__addParent(t1, matmulBackward0)
    if t2.requiresGrad:
        resultTensor.__addParent(t2, matmutBackward1)

    return resultTensor


