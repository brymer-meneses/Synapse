
# Brymer Meneses,
# Start: March 30, 2021

# Decisions
# Tensor.grad must be a Tensor since we will do operations like
# W = W - lr * tensor.grad
#
from typing import Union, Optional, List, NamedTuple, Callable

import numpy as np


Arrayable = Union[float, list, np.ndarray]
Number = Union[float, int]

def ensureArray(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


class Node(NamedTuple):
    tensor: 'Tensor'
    gradfn: Callable[['Tensor'], np.ndarray]

    def __repr__(self):
        return f"[{repr(self.tensor)}, {self.gradfn.__name__}]"


class Tensor:

    def __init__(self, data: Arrayable, requiresGrad: bool = False) -> None:
        from synapse import GradState
        if isinstance(data, (str, Tensor, bool)):
            raise ValueError(f"Passed in unsupported type for Tensor, got: {type(data)}")

        self.__data = ensureArray(data)
        self.shape = self.__data.shape
        self.grad: Optional['Tensor'] = None
        self.parentNodes: List[Node] = []

        if GradState.evalGrad():
            self.requiresGrad = requiresGrad
        else:
            self.requiresGrad = False

        if self.requiresGrad:
            self.zeroGrad()

        return
    @property
    def data(self) -> None:
        return self.__data
    @data.setter
    def data(self, newData) -> None:
        assert isinstance(newData, (np.ndarray, np.float64)), ValueError(f"Expected tensor got, {type(newData)}")
        self.__data = newData
        self.shape = newData.shape
        return

    @property
    def T(self) -> 'Tensor':
        data = self.data.T
        return Tensor(data)

    def __repr__(self):
        return f"<Tensor: {self.shape}, requiresGrad: {self.requiresGrad}>"

    def __str__(self):
        return f"Tensor, requiresGrad={self.requiresGrad} \n{str(self.data)}"

    def addParent(self, node: Node) -> None:
        self.parentNodes.append(node)
        return

    def zeroGrad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))
        return

    def __mul__(self, tensor: 'Tensor') -> 'Tensor':
        from synapse.autograd.__ops import tensorMul
        return tensorMul(self, tensor)

    def __add__(self, tensor: 'Tensor') -> 'Tensor':
        from synapse.autograd.__ops import tensorAdd
        return tensorAdd(self, tensor)

    def __sub__(self, tensor: 'Tensor') -> 'Tensor':
        from synapse.autograd.__ops import tensorAdd, tensorNeg
        return tensorAdd(self, tensorNeg(tensor))

    def __neg__(self) -> 'Tensor':
        from synapse.autograd.__ops import tensorNeg
        return tensorNeg(self)

    def __pow__(self, power: Number) -> 'Tensor':
        from synapse.autograd.__ops import power
        return power(self, power)

    def __matmul__(self, tensor: 'Tensor') -> 'Tensor':
        from synapse.autograd.__ops import matmul
        return matmul(self, tensor)

    def sum(self) -> 'Tensor':
        from synapse.autograd.__ops import tensorSum
        return tensorSum(self)

    def mean(self) -> 'Tensor':
        from synapse.autograd.__ops import tensorMean
        return tensorMean(self)

    def backward(self, grad: 'Tensor' = None) -> None:
        """Executes backpropagation and evaluates
           the gradients of Tensors with
           'requiresGrad = True'. """
        assert self.requiresGrad == True, "Called backwards on a tensor that doesn't require gradients"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-zero tensor")

        self.grad.data = self.grad.data + grad.data #type: ignore

        for node in reversed(self.parentNodes):

            # Calculate the gradient of this node 
            # with respect to the parent node.

            # localGrad represents the derivative 
            # of this tensor with respect to 
            # its parent tensor
            localGrad = node.gradfn(grad.data)

            # Propagate gradients to the each parent 
            # node
            node.tensor.backward(localGrad)

        return









