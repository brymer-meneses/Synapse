
# Lists all the functions used in backpropagation

import numpy as np
from synapse.core.tensor import Tensor
from .types import Number

def powBackward(grad: Tensor, t1: Tensor, power: Number) -> Tensor:
    """Gradient Function that is used when
       tensor.pow(n) or tensor ** n is executed in the
       computation graph

    """
    #print("==== From pow backward ==== ")
    #print("Arguments")
    #print("t1")
    #print(t1.data)
    #print("grad")
    #print(grad.data)
    data = grad.data * np.multiply(power, (t1.data ** (power-1)))
    #print("result")
    #print(data)
    #print("==== End pow backward ==== ")
    return Tensor(data)

def subBackward0(grad: Tensor, t1: Tensor, t2: Tensor) -> Tensor:
    return grad

def subBackward1(grad: Tensor, t1: Tensor, t2: Tensor) -> Tensor:
    data = np.negative(grad.data)
    return Tensor(data)

def meanBackward(grad: Tensor, t1: Tensor) -> Tensor:
    """Gradient Function that is used when
       tensor.mean() is executed in the
       computation graph

    """
    data = np.ones_like(t1.data) / np.size(t1.data)
    data = grad.data * data
    return Tensor(data)

def negBackward(grad: Tensor, t1: Tensor) -> Tensor:
    """Gradient Function that is used when
       - tensor is executed in the
       computation graph

    """

    data = np.negative(grad.data)
    print(data)
    return Tensor(data)

def sumBackward(grad: Tensor, t1: 'Tensor') -> Tensor:
    """Gradient Function that is used when
       tensor.sum() is executed in the
       computation graph


    """

    return Tensor(grad.data *np.ones_like(t1.data))

def addBackward(grad: Tensor, t1: 'Tensor', t2: 'Tensor') -> Tensor:
    """Gradient function that is used when
       a tensor that requires gradient is added
       element wise to another tensor.

       - Math:
           Y = A + B
           dY/dA = 1
           dY/dB = 1
    """

    return grad

def mulBackward0(grad: Tensor, t1: 'Tensor', t2: 'Tensor') -> Tensor:
    """Gradient function that is used when
       a tensor that requires gradient is multiplied
       element-wise to another tensor


       - Math:
           Y = A * B
           dY/dA = B
           dY/dB = A
    """
    return Tensor(grad.data * t2.data)

def mulBackward1(grad: Tensor, t1: 'Tensor', t2: 'Tensor') -> Tensor:
    """Gradient function that is used when
       a tensor that requires gradient is multiplied
       element-wise to another tensor


       - Math:
           Y = A * B
           dY/dA = B
           dY/dB = A
    """
    return Tensor(grad.data * t1.data)

def matmulBackward0(grad: Tensor, t1: 'Tensor', t2: 'Tensor') -> Tensor:
    """Gradient Function that is used when
       a tensor that requires gradient is matrix-multiplied
       to another tensor

       - Math:
          let A.shape == m x n
          let B.shape == n x p

           Y = A @ B
           F(Y) = L
           dF/dA = dF/dY dY/dA

           returns:
                dF/dA = grad * B.T

           where:
                dF/dY -> grad

           returns:
    """
    try:
        result = np.matmul(grad.data, t2.data.T)
    except:
        raise RuntimeError(f"Caught Exception while \
                           trying to perform matrix-multiplication two \
                           matrices with shape: {grad.shape} {t2.data.T.shape}")

    return Tensor(result)


def matmulBackward1(grad: Tensor, t1: 'Tensor', t2: 'Tensor') -> Tensor:
    """Gradient Function that is used when
       a tensor is matrix multiplied to a tensor that
       requires gradient.


       - Math:
          let A.shape == m x n
          let B.shape == n x p

           Y = A @ B
           F(Y) = L
           dF/dB =  (dY/dB).T dF/dY
           returns:
                dF/dB = A.T * grad


           where:
                dF/dY -> grad

    """
    try:
        result = np.matmul(t1.data.T, grad.data)
    except:
        raise RuntimeError(f"Caught Exception while \
                           trying to perform matrix-multiplication two \
                           matrices with shape: {t1.data.T.shape} {grad.shape}")


    return Tensor(result)
