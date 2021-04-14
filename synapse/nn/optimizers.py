
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

        layer.weights.data = layer.weights.data - self.__lr * layer.weights.grad.data
        if layer.use_bias:
            layer.bias.data = layer.bias.data - self.__lr * layer.bias.grad.data

        return


class GDM(Optimizer):
    """Gradient Descent with Momentum"""
    def __init__(self, lr: float=0.001, beta: float =0.9, alpha: float=0.9) -> None:
        self._beta = beta 
        self._alpha = alpha
        self._lr = lr
        self._Vw = 0
        self._Vb = 0

    def step(self, layer: Layer) -> None:
        
        if layer.use_bias:
            self._Vw = self._beta * self._Vw + (1 - self._beta) * self._Vw
            self._Vb = self._beta * self._Vb + (1 - self._beta) * self._Vb

            layer.weights.data = layer.weights.data - self._alpha * self._Vw
            layer.bias.data = layer.bias.data - self._alpha * self._Vb

        else:
            self._Vw = self._beta * self._Vw + (1 - self._beta) * self._Vw

            layer.weights.data = layer.weights.data - self._alpha * self._Vw

        


            
