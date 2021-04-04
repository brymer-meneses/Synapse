
from synapse.autograd.tensor import Tensor
from synapse.nn.layers import Layer
from synapse.nn.optimizers import Optimizer

class Model:

    def __init__(self) -> None:
        raise NotImplementedError

    def __call__(self, input) -> Tensor:
        if not isinstance(input, Tensor):
            raise RuntimeError(f"Expected Tensor received: {type(input)}")

        result = self.forward(input)
        return result

    def forward(self, x) -> Tensor:
        raise NotImplementedError

    def fit(self, x_train: Tensor, y_train: Tensor) -> None:
        raise NotImplementedError

    def summary(self) -> None:
        if not self.__isCompiled:
            raise RuntimeError("Model not compiled")
        print("\n")
        print("==================== Model ====================")

        for layer in self.__layers:
            print(layer)
        print(self.__optimizer)

        print("==============================================")

    def compile(self, optimizer: Optimizer) -> None:
        self.__optimizer = optimizer

        self.__layers = []
        attributes = vars(self)
        for key, value in attributes.items():
            if isinstance(value, Layer):
                self.__layers.append(value)

        self.__isCompiled = True

        return

    def backwards(self, grad: 'Tensor'):

        for layer in reversed(self.__layers):
            grad = layer.backwards(grad)

        self.grad = grad
        return



