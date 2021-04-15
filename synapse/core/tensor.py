
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

def ensure_array(arrayable: Arrayable) -> np.ndarray:
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

    def __init__(self, data: Arrayable, requires_grad: bool = False) -> None:
        if isinstance(data, (str, Tensor, bool)):
            raise ValueError(f"Passed in unsupported type for Tensor, got: {type(data)}")

        self.__data = ensure_array(data)
        self.shape = self.__data.shape
        self.grad: Optional['Tensor'] = None
        self._parent_nodes: List[Node] = []

        from synapse import grad_state

        if grad_state.is_recording:
            self.requires_grad = requires_grad
        else:
            self.requires_grad = False

        if self.requires_grad:
            self.zero_grad()

        return
    @property
    def data(self) -> None:
        return self.__data
    @data.setter
    def data(self, new_data) -> None:
        assert isinstance(new_data, (np.ndarray, np.float64)), ValueError(f"Expected tensor got, {type(new_data)}")
        self.__data = new_data
        self.shape = new_data.shape
        return

    @property
    def T(self) -> 'Tensor':
        data = self.data.T
        return Tensor(data)

    def __repr__(self):
        return f"<Tensor: {self.shape}, requires_grad: {self.requires_grad}>"

    def __str__(self):
        return f"Tensor, requires_grad={self.requires_grad} \n{str(self.data)}"

    def _add_parent(self, node: Node) -> None:
        self._parent_nodes.append(node)
        return

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))
        return

    def __mul__(self, tensor: 'Tensor') -> 'Tensor':
        from synapse.core.ops import mul
        return mul(self, tensor)

    def __add__(self, tensor: 'Tensor') -> 'Tensor':
        from synapse.core.ops import add
        return add(self, tensor)

    def __sub__(self, tensor: 'Tensor') -> 'Tensor':
        from synapse.core.ops import sub
        return sub(self, tensor)

    def __neg__(self) -> 'Tensor':
        from synapse.core.ops import neg
        return neg(self)

    def __pow__(self, power: Number) -> 'Tensor':
        from synapse.core.ops import pow
        return pow(self, power)

    def __matmul__(self, tensor: 'Tensor') -> 'Tensor':
        from synapse.core.ops import matmul
        return matmul(self, tensor)

    def sum(self) -> 'Tensor':
        from synapse.core.ops import sum
        return sum(self)

    def mean(self) -> 'Tensor':
        from synapse.core.ops import mean
        return mean(self)

    def backward(self, grad: 'Tensor' = None) -> None:
        """Executes backpropagation and evaluates
           the gradients of Tensors with
           'requiresGrad = True'. """
        assert self.requires_grad == True, "Called backwards on a tensor that doesn't require gradients"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-zero tensor")

        self.grad.data = self.grad.data + grad.data #type: ignore

        for node in self._parent_nodes:

            # Calculate the gradient of this node 
            # with respect to the parent node.

            # local_grad represents the derivative 
            # of this tensor with respect to 
            # its parent tensor
            #print(f"\ntransforming {grad}, using: ")
            local_grad = node.gradfn(grad)
            #print(f"{node.gradfn.__name__}")
            #print(localGrad)

            # Propagate gradients to the each parent 
            # node
            node.tensor.backward(local_grad)

        return









