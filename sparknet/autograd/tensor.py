
# Brymer Meneses,
# Start: March 30, 2021

from typing import Union, Optional, List, NamedTuple, Callable

import numpy as np

from ops import tensorSum

Arrayable = Union[float, list, np.ndarray]


def ensureArray(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)
    return


class Node(NamedTuple):
    tensor: 'Tensor'
    gradfn = Callable[[np.ndarray], np.ndarray]


class Tensor:
    grad: Optional[np.ndarray] = None
    parentNodes: List[Node] = []

    def __init__(self, data: Arrayable, requiresGrad: bool = False):
        self.data = ensureArray(data)

    def __addParent(self, node: Node):
        self.parentNodes.append(node)
        return

    def sum(self) -> 'Tensor':
        return tensorSum(self)
