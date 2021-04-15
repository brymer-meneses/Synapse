from typing import Union

from synapse.core.tensor import Tensor
from synapse.nn.layers import Layer
from synapse.nn.optimizers import Optimizer
from synapse.core.differentiable import Differentiable

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

    def zero_grad(self) -> None:
        for _, layer in self.layers:
            layer.zero_grad()
        return

    def fit(self, x_train: Tensor, y_train: Tensor, epochs: int) -> None:
        totalLoss = 0
        for epoch in range(epochs):
            epoch_loss = 0
            output = self.forward(x_train)

            output_loss = self._loss(output, y_train)
            output_loss.backward()

            self.optimize()

            epoch_loss += output_loss.data
            total_loss += epoch_loss

            print(f"{epoch} {epoch_loss}")

        print("Finished Training ",total_loss)

    def summary(self) -> None:
        if not self.__is_compiled:
            raise RuntimeError("Model not compiled")
        print("\n")
        print("=================== Model =====================")

        attributes = vars(self)
        for key, value in attributes.items():
            if isinstance(value, Layer):
                print(value)

        print(self._optimizer)
        print(self._loss.__name__) 

        print("==============================================")

    def compile(self, optimizer: Optimizer, loss: Differentiable) -> None:
        #assert isinstance(optimizer, Optimizer), ValueError(f"Expected Optimizer received {type(optimizer)}")
        #assert isinstance(loss, (Differentiable)), ValueError(f"Expected TensorFunction received {type(loss)}")

        self.__optimizer = optimizer
        self.__loss = loss

        self.layers = []
        attributes = vars(self)

        for key, value in attributes.items():
            if isinstance(value, Layer):
                self.layers.append((key, value))

        self.__is_compiled = True

    def optimize(self) -> None:
        for _, layer in self.layers:
            if isinstance(layer, Layer):
                self.__optimizer.step(layer)

        return




