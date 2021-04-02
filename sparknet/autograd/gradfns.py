
# Lists all the functions used in backpropagation

import numpy as np

def sumBackward(grad: np.ndarray, t1: 'Tensor', t2: 'Tensor') -> np.ndarray:
    """Gradient Function that is used when
       tensor.sum() is executed in the
       computation graph


    """

    return grad*np.ones_like(t1.data)

def addBackward(grad: np.ndarray, t1: 'Tensor', t2: 'Tensor') -> np.ndarray:
    """Gradient function that is used when
       a tensor that requires gradient is added
       element wise to another tensor.

       - Math:
           Y = A + B
           dY/dA = 1
           dY/dB = 1
    """

    return grad

def mulBackward(grad: np.ndarray, t1: 'Tensor', t2: 'Tensor') -> np.ndarray:
    """Gradient function that is used when
       a tensor that requires gradient is multiplied
       element-wise to another tensor


       - Math:
           Y = A * B
           dY/dA = B
           dY/dB = A
    """
    return grad * t2.data

def matmulBackward0(grad: np.ndarray, t1: 'Tensor', t2: 'Tensor') -> np.ndarray:
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
    return np.matmul(grad, t2.data.T)


def matmulBackward1(grad: np.ndarray, t1: 'Tensor', t2: 'Tensor') -> np.ndarray:
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

    return np.matmul(t1.data.T, grad)
