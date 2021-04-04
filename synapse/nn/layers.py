
from synapse.autograd.tensor import Tensor
import numpy as np

class Layer:
    def __init__(self, in_features: int ,out_features: int) -> None:
        raise NotImplementedError

    def __call__(self, x: "Tensor") -> 'Tensor':
        raise NotImplementedError

    def backwards(self, grad: 'Tensor') -> 'Tensor':
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_features: int ,out_features: int, useBias=False) -> None:
        self.useBias = useBias

        self.weights = Tensor(
            data=np.random.randn(out_features, in_features),
            requiresGrad=True)

        if useBias:

            # self.bias must handle broadcasting
            # but idk how to implement it
            self.bias = Tensor(
                np.random.randn(out_features, 1),
                requiresGrad=True)

    def __call__(self, x: 'Tensor') -> 'Tensor':
        """Forward Propagation"""

        if self.useBias:
            self._output = self.weights @ x + self.bias
        else:
            self._output = self.weights @ x

        # self.weights.shape == out x in
        # x.shape == in x num
        return self._output

    def backwards(self, grad: 'Tensor') -> 'Tensor':
        self._output.backwards(grad)
        return self._output.grad




