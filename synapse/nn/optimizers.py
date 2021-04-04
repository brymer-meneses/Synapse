
from synapse import Tensor
from synapse.nn.layers import Layer
import synapse as sn

class Optimizer:
    def __init__(self, lr: float = 0.001) -> None:
        self.__lr = lr
        return

    def step(self) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr: float =0.001) -> None:
        self.__lr = lr

    def __str__(self) -> str:
        return f'\n=> Optimizer: SGD \
            \n\t- lr {self.__lr}'

    def step(self, layer: Layer) -> None:

        layer.weights.data = layer.weights.data - self.lr * layer.weights.grad.data
        if layer.useBias:
            layer.bias.data = layer.bias.data - self.lr * layer.bias.grad.data

        return
