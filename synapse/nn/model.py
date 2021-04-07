from typing import Union

from synapse import TensorFunction, TensorBinaryFunction
from synapse.autograd.tensor import Tensor
from synapse.nn.layers import Layer
from synapse.nn.optimizers import Optimizer

class Model:

    def __init__(self) -> None:
        raise NotImplementedError

    def __call__(self, input) -> Tensor:
        if not isinstance(input, Tensor):
            raise ValueError(f"Expected Tensor received: {type(input)}")

        result = self.forward(input)
        return result

    def forward(self, x) -> Tensor:
        raise NotImplementedError

    def zeroGrad(self) -> None:
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer.weights.zeroGrad()
        return

    def fit(self, x_train: Tensor, y_train: Tensor, epochs: int) -> None:
        totalLoss = 0
        for epoch in range(epochs):
            epochLoss = 0
            output = self.forward(x_train)

            outputLoss = self.__loss(output, y_train)
            outputLoss.backward()

            self.optimize()

            epochLoss += outputLoss.data
            totalLoss += epochLoss

            print(f"{epoch} {epochLoss}")

        print("Finished Training ",totalLoss)

    def summary(self) -> None:
        if not self.__isCompiled:
            raise RuntimeError("Model not compiled")
        print("\n")
        print("=================== Model =====================")

        attributes = vars(self)
        for key, value in attributes.items():
            if isinstance(value, (Layer, TensorFunction, TensorBinaryFunction)):
                print(value)

        print(self.__optimizer)

        print("==============================================")

    def compile(self, optimizer: Optimizer, loss: TensorFunction) -> None:
        assert isinstance(optimizer, Optimizer), ValueError(f"Expected Optimizer received {type(optimizer)}")
        assert isinstance(loss, TensorFunction), ValueError(f"Expected TensorFunction received {type(loss)}")

        self.__optimizer = optimizer
        self.__loss = loss

        self.layers = []
        attributes = vars(self)
        for key, value in attributes.items():
            if isinstance(value, Layer):
                self.layers.append((key, value))

        self.__isCompiled = True

    def optimize(self) -> None:
        for _, layer in self.layers:
            if isinstance(layer, Layer):
                self.__optimizer.step(layer)

        return




