
from sparknet.nn.layers import Layer
from sparknet.autograd.tensor import Tensor
import numpy as np

# Note: 
# All activation layers inherit from the "Layer" base class

# TODO NOT COMPLETE

class Tanh(Layer):
    def __init__(self):
        return

    def forward(self, x: 'Tensor') -> 'Tensor':
        result = np.tanh(x.data)
        self.inputs = x
        return result

    def backwards(self, grad: 'Tensor') -> 'Tensor':
        return

class ReLU(Layer):
    def __init__(self):
        return

    def forward(self, x: 'Tensor') -> 'Tensor':
        result = Tensor(np.where(x.data < 0, 0, x.data), requiresGrad=True)
        return result

    def backwards(self, grad: 'Tensor') -> 'Tensor':
        # TODO implement backpropagation for ReLU
        return


