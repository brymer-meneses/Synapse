from typing import List
from ._types import GradFn

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

class Differentiable(object):
    from synapse.autograd.tensor import Tensor

    def __init__(self, function, gradFn0, gradFn1=None):
        self.function = function
        self.gradFn0 = gradFn0
        self.gradFn1 = gradFn1

    def __call__(self, *args) -> Tensor:
        from synapse.autograd.tensor import Node
        resultTensor = self.function(*args)

        tensors = _getTensorArgs(*args)

        # Handle when len(tensors) is not possible
        # because there were no tensors passed.
        if len(tensors) == 0:
            raise ValueError("Expected tensor argument got None")

        elif len(tensors) > 2:
            raise ValueError(f"Maximum number of tensor arguments is 2 got {len(tensors)}")

        elif len(tensors) == 1:
            if tensors[0].requiresGrad:
                node = Node(tensors[0], _package(self.gradFn0, *args))
                resultTensor.addParent(node)

        # Check if it is a binary tensor function
        elif len(tensors) == 2:
            if tensors[0].requiresGrad:
                node = Node(tensors[0], _package(self.gradFn0, *args))
                resultTensor.addParent(node)

            if tensors[1].requiresGrad:
                if self.gradFn1 is None:
                    # Defaults to the first gradient function if 
                    # two tensors are passed and gradFn1 is not 
                    # provided.
                    # This is useful in the case of the tensor add 
                    # function, which has the same gradient irrespective 
                    # of the tensor arguments
                    node = Node(tensors[1], _package(self.gradFn0, *args))
                else:
                    node = Node(tensors[1], _package(self.gradFn1, *args))
                resultTensor.addParent(node)


        return resultTensor



