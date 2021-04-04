
from synapse.autograd.tensor import Tensor
from synapse.nn.layers import Layer

class Model:

    def __init__(self):
        raise NotImplementedError

    def __call__(self, input) -> Tensor:
        if not isinstance(input, Tensor):
            raise RuntimeError(f"Expected Tensor received: {typeof(input)}")

        result = self.forward(input)
        return result

    def forward(self, x) -> Tensor:
        raise NotImplementedError

    def fit(self, x_train, y_train):
        raise NotImplementedError

    def backwards(self, grad: 'Tensor'):
        layers = []
        attributes = vars(self)

        for key, value in attributes.items():
            if isinstance(value, Layer):
                layers.append(value)

        for layer in reversed(layers):
            grad = layer.backwards(grad)

        self.grad = grad
        return



