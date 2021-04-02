
# Lists all the functions used in backpropagation

import numpy as np

def sumBackward(tensor: 'Tensor', grad: np.ndarray) -> np.ndarray:
    """Gradient Function that is used when
       tensor.sum() is executed in the
       computation graph

       Paremeters:
            tensor: Tensor
            grad: np.ndarray

    """

    return grad*np.ones_like(tensor.data)

def addBackward(grad: np.ndarray) -> np.ndarray:
    """Gradient function that is used when
       a tensor that requires gradient is added
       element wise to another tensor.

       - Math:
           Y = A + B
           dY/dA = 1
           dY/dB = 1
    """

    return grad

def mulBackward(otherTensorData: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """Gradient function that is used when
       a tensor that requires gradient is multiplied
       element-wise to another tensor

       - Paremeters:
            tensor: Tensor
            grad: np.ndarray

       - Math:
           Y = A * B
           dY/dA = B
           dY/dB = A
    """
    return grad * otherTensorData

def matmulBackward0(grad: np.ndarray, t2Data: np.ndarray) -> np.ndarray:
    """Gradient Function that is used when
       a tensor that requires gradient is matrix-multiplied
       to another tensor
       - Paremeters:
            t2Data: np.ndarray
            grad: np.ndarray

       - Math:
          let A.shape == m x n
          let B.shape == n x p

           Y = A @ B
           F(Y) = L
           dF/dA = dF/dY dY/dA

           returns:
                dF/dA = grad * A.T

           where:
                dF/dY -> grad

           returns:
    """
    return np.matmul(grad, t2Data.T)


def matmulBackward1(grad: np.ndarray, t2Data: np.ndarray) -> np.ndarray:
    """Gradient Function that is used when
       a tensor is matrix multiplied to a tensor that
       requires gradient.

       - Paremeters:
            tensor: Tensor
            grad: np.ndarray

       - Math:
          let A.shape == m x n
          let B.shape == n x p

           Y = A @ B
           F(Y) = L
           dF/dB =  (dY/dB).T dF/dY
           returns:
                dF/dB = B.T * grad


           where:
                dF/dY -> grad

    """

    return np.matmul(B.T, grad)
