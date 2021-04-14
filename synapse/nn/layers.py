from abc import ABC, abstractmethod
from synapse.autograd.tensor import Tensor

import numpy as np

class Layer(ABC):
    @abstractmethod
    def __init__(self,
                 inFeatures: int ,
                 outFeatures: int,
                 useBias: bool =False,
                 name: str = "Layer" ) -> None:

        raise NotImplementedError

    def __call__(self, x: "Tensor") -> 'Tensor':
        self.__output = self.forward(x)
        return self.__output

    def zeroGrad(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: "Tensor") -> "Tensor":
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError



class Linear(Layer):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 use_bias: bool =False,
                 name: str = 'Linear') -> None:

        self._use_bias = use_bias
        self._in_features = in_features
        self._out_features = out_features
        self._name = name

        self.weights = Tensor(
            data=np.random.randn(out_features, in_features),
            requires_grad=True)

        if use_bias:

            # self.bias must handle broadcasting
            # but idk how to implement it
            self.bias = Tensor(
                np.random.randn(out_features, 1),
                requires_grad=True)

    def forward(self, x: 'Tensor') -> 'Tensor':
        """Forward Propagation"""

        assert self.weights.shape[-1] == x.shape[0], \
        f"""\nThe shape is not compatible with the this layers in features\n
        Expected: ({self.weights.shape[-1]}, x) Got: ({x.shape[0]}, x)"""

        if self._use_bias:
            output = self.weights @ x + self.bias
        else:
            output = self.weights @ x

        return output

    def __str__(self) -> str:
        return f'{self__name} = ({self._in_features}, {self._out_features})'

    def zero_grad(self) -> None:
        self.weights.zero_grad()
        if self.useBias:
            self.bias.zero_grad()
        return






