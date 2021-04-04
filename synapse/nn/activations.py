
from sparknet.nn.layers import Layer
from sparknet.autograd.tensor import Tensor
import numpy as np

# Note: 
# All activation layers inherit from the "Layer" base class

# TODO NOT COMPLETE


class Activation(Layer):

    def __init__(self):
        return

    def __call__(self, input: 'Tensor') -> 'Tensor':
        if not isinstance(input, Tensor):
            raise RuntimeError(f"Expected type Tensor got: {type(input)}")

        return self.forward(input)

    def function(self, input: np.ndarray) -> np.ndarray:
        """y = f(x)"""
        raise NotImplementedError

    def functionPrime(self, input: np.ndarray) -> np.ndarray:
        """y' = f'(x)"""
        raise NotImplementedError


    def forward(self, x: 'Tensor') -> 'Tensor':
        self.inputs = x
        result = Tensor(self.function(x.data), requiresGrad=True)
        return result

    def backwards(self, grad: 'Tensor') -> 'Tensor':
        grad = Tensor(self.functionPrime(self.inputs) * grad.data, requiresGrad=True)
        return grad



class Tanh(Activation):
    def function(self, input: np.ndarray) -> np.ndarray:
        result = np.tanh(input)
        return result

    def functionPrime(self, input: np.ndarray) -> np.ndarray:
        y = self.function(input)
        return 1 - y**2


class ReLU(Activation):
    def function(self, input: np.ndarray) -> np.ndarray:
        result = np.where(input <0, 0, input)
        return result

    def functionPrime(self, input: np.ndarray) -> np.ndarray:
        result = np.where(input < 0 | input == 0, 0, 1)
        return result

