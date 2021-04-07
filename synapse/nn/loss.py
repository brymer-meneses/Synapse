from synapse import Tensor, TensorFunction
from synapse.autograd.tensor import Node
import synapse as sn
from abc import abstractmethod

import numpy as np
from typing import Callable

class Loss(TensorFunction):
    def __call__(self, predicted: Tensor, actual: Tensor) -> Tensor:
        resultTensor = self.forward(predicted, actual)
        if predicted.requiresGrad:
            node = Node(predicted, self.gradFn(predicted, actual))
            resultTensor.addParent(node)
        return resultTensor

    @abstractmethod
    def forward(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return

    @abstractmethod
    def gradFn(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return





class MSE(Loss):


    def __str__(self) -> str:
        return "Mean Squared Error"

    def forward(self, predicted: Tensor, actual: Tensor) -> Tensor:
        resultTensor = sn.power(predicted - actual, 2).mean()

        return resultTensor

    def gradFn(self, predicted: Tensor, actual: Tensor) -> Callable[[np.ndarray], Tensor]:

        def mseBackward(grad: np.ndarray) -> Tensor:
            data = grad * (2 * np.subtract(predicted.data, actual.data)).mean()
            resultTensor = Tensor(data)
            return resultTensor

        return mseBackward




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






