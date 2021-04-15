from typing import List, Callable
from .types import GradFn
from functools import wraps

def _package(gradfn, *args) -> GradFn:
    from synapse import Tensor

    def packaged_gradfn(grad: Tensor) -> Tensor:
        return gradfn(grad, *args)

    # Changes the name of packagedGradFn to gradfn
    # this makes the debugging process much easier 
    # especially if we want to look at the computation
    # graph.
    packaged_gradfn.__name__ = gradfn.__name__

    return packaged_gradfn

def _get_tensor_args(*args) -> List['Tensor']:
    from synapse import Tensor
    result = []
    for arg in args:
        if isinstance(arg, Tensor):
            result.append(arg)
    return result

def Differentiable(gradFn0: Callable, gradFn1: Callable=None) -> Callable:

    def decorator(TensorFunction: Callable) -> Callable:
        from synapse.core.tensor import Node, Tensor

        @wraps(TensorFunction)
        def wrapper(*args) -> Tensor:
            tensor_args = _get_tensor_args(*args)
            total_tensor_args = len(tensor_args)

            result_tensor = TensorFunction(*args)

            if total_tensor_args == 0:
                raise ValueError(f"No Tensor arguments were passed to {function.__name__}")

            elif total_tensor_args > 2:
                raise ValueError(f"Expected < 2 tensor arguments got {total_tensor_args}")

            elif total_tensor_args == 1:
                t1 = tensor_args[0]

                if t1.requires_grad:
                    node = Node(t1, _package(gradFn0, *args))
                    result_tensor._add_parent(node)
            else:
                t1, t2 = tensor_args

                if t1.requires_grad:
                    node = Node(t1, _package(gradFn0, *args))
                    result_tensor._add_parent(node)

                if t2.requires_grad:
                    if gradFn1 is None:
                        # Defaults to the first gradient function if 
                        # two tensors are passed and gradFn1 is not 
                        # provided.
                        # This is useful in the case of the tensor add 
                        # function, which has the same gradient irrespective 
                        # of the tensor arguments
                        node = Node(t2, _package(gradFn0, *args))
                    else:
                        node = Node(t2, _package(gradFn1, *args))
                    result_tensor._add_parent(node)

            return result_tensor
        return wrapper

    return decorator



