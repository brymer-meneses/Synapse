from typing import List, Callable
from ._types import GradFn
from functools import wraps

def _package(gradfn, *args) -> GradFn:
    from synapse import Tensor

    def packagedGradFn(grad: Tensor) -> Tensor:
        return gradfn(grad, *args)

    # Changes the name of packagedGradFn to gradfn
    # this makes the debugging process much easier 
    # especially if we want to look at the computation
    # graph.
    packagedGradFn.__name__ = gradfn.__name__

    return packagedGradFn

def _getTensorArgs(*args) -> List['Tensor']:
    from synapse import Tensor
    result = []
    for arg in args:
        if isinstance(arg, Tensor):
            result.append(arg)
    return result

def Differentiable(gradFn0: Callable, gradFn1: Callable=None) -> Callable:

    def decorator(TensorFunction: Callable) -> Callable:
        from synapse.autograd.tensor import Node, Tensor

        @wraps(TensorFunction)
        def wrapper(*args) -> Tensor:
            tensorArgs = _getTensorArgs(*args)
            totalTensorArgs = len(tensorArgs)

            resultTensor = TensorFunction(*args)

            if totalTensorArgs == 0:
                raise ValueError(f"No Tensor arguments were passed to {function.__name__}")

            elif totalTensorArgs > 2:
                raise ValueError(f"Expected < 2 tensor arguments got {totalTensorArgs}")

            elif totalTensorArgs == 1:
                t1 = tensorArgs[0]

                if t1.requiresGrad:
                    node = Node(t1, _package(gradFn0, *args))
                    resultTensor._addParent(node)
            else:
                t1, t2 = tensorArgs

                if t1.requiresGrad:
                    node = Node(t1, _package(gradFn0, *args))
                    resultTensor._addParent(node)

                if t2.requiresGrad:
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
                    resultTensor._addParent(node)

            return resultTensor
        return wrapper

    return decorator



