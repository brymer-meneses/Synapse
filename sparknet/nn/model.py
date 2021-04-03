
from sparknet.autograd.tensor import Tensor
from sparknet.nn.layers import Layer

class Model:

    def __init__(self):
        raise NotImplementedError

    def __call__(self, input):
        if not isinstance(input, Tensor):
            raise RuntimeError(f"Expected Tensor received: {typeof(input)}")

        self.forward(input)
        return

    def forward(self, x):
        raise NotImplementedError

    def fit(self, x_train, y_train):
        raise NotImplementedError

    def backward(self, grad: 'Tensor'):
        layers = []
        attributes = vars(self)

        for key, value in attributes.items():
            if isinstance(value, Layer):
                layers.append(value)

        for layer in reversed(layers):
            grad = layer.backwards(grad)

        self.grad = grad
        return



