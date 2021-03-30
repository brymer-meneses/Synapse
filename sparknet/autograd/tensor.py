
# Brymer Meneses,
# Start: March 30, 2021

# Decisions
# Tensor.grad must be a Tensor since we will do operations like
# W = W - lr * tensor.grad
#
from typing import Union, Optional, List, NamedTuple, Callable

import numpy as np

from ops import tensorSum, tensorAdd

Arrayable = Union[float, list, np.ndarray]

def ensureArray(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)


class Node(NamedTuple):
    tensor: 'Tensor'
    gradfn = Callable[['Tensor'], 'Tensor']


class Tensor:

    def __init__(self, data: Arrayable, requiresGrad: bool = False) -> None:
        self.data = ensureArray(data)
        self.requiresGrad = requiresGrad
        self.shape = data.shape
        self.grad: Optional['Tensor'] = None
        self.parentNodes: List[Node] = []

        if requiresGrad:
            self.zeroGrad()

        return

    def __addParent(self, node: Node) -> None:
        self.parentNodes.append(node)
        return

    def zeroGrad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))
        return

    def __add__(self, tensor: 'Tensor') -> 'Tensor':
        return tensorAdd(self, tensor)

    def sum(self) -> 'Tensor':
        return tensorSum(self)

    def backwards(self, grad: 'Tensor' = None) -> None:
        """Executes backpropagation and evaluates
           the gradients of Tensors with
           'requiresGrad = True'. """

        if grad is None:
            if self.shape == ():
                self.grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-zero tensor")

        self.grad.data = self.grad.data + grad.data

        for node in self.parentNodes:

            # Calculate the gradient of this node 
            # with respect to the parent node.

            # localGrad represents the derivative 
            # of this tensor with respect to 
            # its parent tensor
            localGrad = Tensor(node.gradfn(grad))

            # Propagate gradients to the each parent 
            # node
            node.tensor.backwards(localGrad)

        return









