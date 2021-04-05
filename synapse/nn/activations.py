
from synapse.nn.layers import Layer
from synapse.autograd.tensor import Tensor

import numpy as np

class Activation(Operation):

    def operation(self, t1: "Tensor", t2: "Tensor") -> "Tensor":
        raise NotImplementedError

    def operationPrime(self, grad: np.ndarray, t1: "Tensor", t2: "Tensor") -> np.ndarray:
        raise NotImplementedError

class Tanh(Activation):

    def operation(self, t1: "Tensor") -> "Tensor":
        data = np.tanh(t1.data)
        requiresGrad = t1.requiresGrad
        resultTensor = Tensor(data, requiresGrad)

        return

    def operationPrime(self, grad: )





