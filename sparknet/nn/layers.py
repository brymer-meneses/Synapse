
from sparknet.autograd.tensor import Tensor
import numpy as np

class Layer:
    def __init__(self, in_features: int ,out_features: int) -> None:
        raise NotImplementedError

    def __call__(self, x: "Tensor") -> 'Tensor':
        raise NotImplementedError

    def backwards(self, grad: 'Tensor') -> 'Tensor':
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_features: int ,out_features: int) -> None:

        self.weights = Tensor(
            np.random.randn(in_features, out_features),
            requiresGrad=True)

        self.bias = Tensor(
            np.random.randn(in_features, out_features),
            requiresGrad=True)

    def __call__(self, x: 'Tensor') -> 'Tensor':
        """Forward Propagation"""
        self._output = self.weights @ x + self.bias
        return self._output

    def backwards(self, grad: 'Tensor') -> 'Tensor':
        self._output.backwards(grad)
        return self._output.grad




