
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
        self.data = ensureArray(data)
        self.requiresGrad = requiresGrad
        self.shape = self.data.shape
        self.grad: Optional['Tensor'] = None
        self.parentNodes: List[Node] = []

        if requiresGrad:
            self.zeroGrad()

        return
    def __repr__(self):
        return f"<Tensor: {self.shape}, requiresGrad: {self.requiresGrad}>"

    def addParent(self, node: Node) -> None:
        self.parentNodes.append(node)
        return

    def zeroGrad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))
        return

    def __mul__(self, tensor: 'Tensor') -> 'Tensor':
        from sparknet.autograd.ops import tensorMul
        return tensorMul(self, tensor)

    def __add__(self, tensor: 'Tensor') -> 'Tensor':
        from sparknet.autograd.ops import tensorAdd
        return tensorAdd(self, tensor)

    def __matmul__(self, tensor: 'Tensor') -> 'Tensor':
        from sparknet.autograd.ops import tensorMatMul
        return tensorMatMul(self, tensor)

    def sum(self) -> 'Tensor':
        from sparknet.autograd.ops import tensorSum
        return tensorSum(self)

    def backwards(self, grad: 'Tensor' = None) -> None:
        """Executes backpropagation and evaluates
           the gradients of Tensors with
           'requiresGrad = True'. """
        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-zero tensor")

        self.grad.data = self.grad.data + grad.data #type: ignore

        for node in self.parentNodes:

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








