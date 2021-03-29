
# Lists all the functions used in backpropagation

import numpy as np


def sumBackward(grad: np.ndarray) -> np.ndarray:
    """Gradient Function that is used when 
       tensor.sum() is executed in the 
       computation graph"""
    return np.ones_like(grad)
