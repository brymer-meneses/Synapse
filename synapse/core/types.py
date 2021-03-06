from synapse.core.tensor import Tensor
from typing import Union, Callable

Number = Union[int, float]
GradFn = Callable[['Tensor'], 'Tensor']

