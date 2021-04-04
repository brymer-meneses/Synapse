
# Brymer Meneses,
# Start: March 30, 2021

# Decisions
# Tensor.grad must be a Tensor since we will do operations like
# W = W - lr * tensor.grad
#
from typing import Union, Optional, List, NamedTuple, Callable

import numpy as np


Arrayable = Union[float, list, np.ndarray]

def ensureArray(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


class Node(NamedTuple):
    tensor: 'Tensor'
    gradfn: Callable[['Tensor'], np.ndarray]


class Tensor:

    def __init__(self, data: Arrayable, requiresGrad: bool = False) -> None:
        from synapse import GradState

        self.data = ensureArray(data)
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None
        self.parentNodes: List[Node] = []

        if GradState.evalGrad():
            self.requiresGrad = requiresGrad
        else:
            self.requiresGrad = False

        if self.requiresGrad:
            self.zeroGrad()

        return
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
        from synapse.autograd.ops import tensorMul
        return tensorMul(self, tensor)

    def __add__(self, tensor: 'Tensor') -> 'Tensor':
        from synapse.autograd.ops import tensorAdd
        return tensorAdd(self, tensor)

    def __matmul__(self, tensor: 'Tensor') -> 'Tensor':
        from synapse.autograd.ops import matmul
        return matmul(self, tensor)

    def sum(self) -> 'Tensor':
        from synapse.autograd.ops import tensorSum
        return tensorSum(self)

    def backwards(self, grad: 'Tensor' = None) -> None:
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
            localGrad = Tensor(node.gradfn(grad.data))

            # Propagate gradients to the each parent 
            # node
            node.tensor.backwards(localGrad)

        return









