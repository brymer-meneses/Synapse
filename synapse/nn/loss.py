from synapse import Tensor, TensorFunction
from synapse.autograd.tensor import Node
from abc import abstractmethod

import numpy as np
from typing import Callable

class Loss(TensorFunction):
    def __call__(self, predicted: Tensor, actual: Tensor) -> Tensor:
        self.function(predicted, actual)
        if predicted.requiresGrad:
            node = Node(predicted, self.gradFn0(predicted))
        return

    @abstractmethod
    def forward(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return

    @abstractmethod
    def gradFn0(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return





class MSE(TensorFunction):
    def __call__(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return self.function(predicted, actual)

    def __str__(self) -> str:
        return "Mean Squared Error"

    def forward(self, predicted: Tensor, actual: Tensor) -> Tensor:
        requiresGrad = predicted.requiresGrad
        data = np.square(np.subtract(predicted.data, actual.data)).mean()

        return Tensor(data, requiresGrad)

    def gradFn(self, predicted: Tensor, actual: Tensor) -> Callable[[np.ndarray], Tensor]:

        def mseBackward(grad: np.ndarray) -> Tensor:
            data = 2 * np.subtract(predicted.data, actual.data).mean()
            resultTensor = Tensor(data)
            return resultTensor

        return




class CategoricalCrossEntropy(TensorFunction):
    """TODO"""
    def forward(self, x, y) -> Tensor:
        if np.any(x < 0) or np.any(y < 0):
            raise ValueError("Only non-zero values are allowed as an input to this function")

        x = np.array(x, dtype=np.float)
        y = np.array(y, dtype=np.float)
        x /= np.sum(x)
        y /= np.sum(y)

        mask = y > 0
        y = y[mask]
        x = x[mask]

        result = -np.sum(x*np.log(y))

        return Tensor(result, x.requiresGrad)






