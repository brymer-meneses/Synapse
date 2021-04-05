from abc import ABC, abstractmethod
from typing import Callable


class TensorFunction(ABC):
    from synapse.autograd.tensor import Tensor, Node
    def __call__(self, t1: Tensor):
        from synapse.autograd.tensor import Node

        resultTensor = self.function(t1)

        if t1.requiresGrad:
            node = Node(t1, self.gradFn0(t1))
            resultTensor.addParent(node)

        return resultTensor

    @abstractmethod
    def function(self, t1: Tensor) -> Tensor:
        pass

    @abstractmethod
    def gradFn0(self, t1: Tensor) -> Tensor:
        pass

class TensorBinaryFunction(ABC):
    from synapse.autograd.tensor import Tensor, Node
    def __call__(self, t1: Tensor, t2: Tensor):
        from synapse.autograd.tensor import Node


        resultTensor = self.function(t1, t2)

        if t1.requiresGrad:
            node = Node(t1, self.gradFn0(t1, t2))
            resultTensor.addParent(node)
        if t2.requiresGrad:
            node = Node(t1, self.gradFn1(t1, t2))
            resultTensor.addParent(node)

        return resultTensor

    @abstractmethod
    def function(self, t1: Tensor, t2: Tensor) -> Tensor:
        pass

    @abstractmethod
    def gradFn0(self, t1: Tensor, t2: Tensor) -> Callable[[Tensor], Tensor]:
        pass

    @abstractmethod
    def gradFn1(self, t1: Tensor, t2: Tensor) -> Callable[[Tensor], Tensor]:
        pass



