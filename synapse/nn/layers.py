
from synapse.autograd.tensor import Tensor
import numpy as np

class Layer:
    def __init__(self,
                 inFeatures: int ,
                 outFeatures: int,
                 useBias: bool =False,
                 name: str = "Layer" ) -> None:

        raise NotImplementedError

    def __call__(self, x: "Tensor") -> 'Tensor':
        self.__output = self.forward(x)
        return self.__output

    def forward(self, x: "Tensor") -> "Tensor":
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    def backward(self, grad: 'Tensor') -> None:
        self.__output.backward(grad)
        return


class Linear(Layer):
    def __init__(self,
                 inFeatures: int,
                 outFeatures: int,
                 useBias: bool =False,
                 name: str = 'Linear') -> None:

        self.useBias = useBias
        self.__inFeatures = inFeatures
        self.__outFeatures = outFeatures
        self.__name = name

        self.weights = Tensor(
            data=np.random.randn(outFeatures, inFeatures),
            requiresGrad=True)

        if useBias:

            # self.bias must handle broadcasting
            # but idk how to implement it
            self.bias = Tensor(
                np.random.randn(out_features, 1),
                requiresGrad=True)

    def forward(self, x: 'Tensor') -> 'Tensor':
        """Forward Propagation"""

        if self.useBias:
            output = self.weights @ x + self.bias
        else:
            output = self.weights @ x

        return output

    def __str__(self) -> str:
        return f'{self.__name} = ({self.__inFeatures}, {self.__outFeatures})'






