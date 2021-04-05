
from typing import Callable
import numpy as np
from .__tensorFunctions import TensorFunction, TensorBinaryFunction


class Sum(TensorFunction):
    from synapse.autograd.tensor import Tensor, Node

    def function(self, t1: Tensor) -> Tensor:
        from synapse.autograd.tensor import Tensor
        data = np.sum(t1.data)
        requiresGrad = t1.requiresGrad
        resultTensor = Tensor(data, requiresGrad)

        return resultTensor


    def gradFn0(self, t1: Tensor) -> Tensor:
        from synapse.autograd.gradfns import sumBackward
        from synapse.autograd.tensor import Tensor

        def SumBackward(grad: Tensor) -> Callable[[np.ndarray, Tensor], Tensor]:
            return sumBackward(grad.data, t1)

        return SumBackward

class Add(TensorBinaryFunction):
    from synapse.autograd.tensor import Tensor, Node

    def function(self, t1: Tensor, t2: Tensor) -> Tensor:
        from synapse.autograd.tensor import Tensor
        if not t1.shape == t2.shape:
            raise RuntimeError("Broadcasting not Implemented (yet)")
        else:
            data = t1.data + t2.data

        requiresGrad = t1.requiresGrad or t2.requiresGrad
        resultTensor = Tensor(data, requiresGrad)

        return resultTensor


    def gradFn0(self, t1: Tensor, t2: Tensor) -> Tensor:
        from synapse.autograd.gradfns import addBackward
        from synapse.autograd.tensor import Tensor

        def AddBackward(grad: Tensor) -> Callable[[np.ndarray, Tensor, Tensor], Tensor]:
            return addBackward(grad.data, t1, t2)

        return AddBackward

    def gradFn1(self, t1: Tensor, t2: Tensor) -> Tensor:
        from synapse.autograd.gradfns import addBackward
        from synapse.autograd.tensor import Tensor

        def AddBackward(grad: Tensor) -> Callable[[np.ndarray, Tensor, Tensor], Tensor]:
            return addBackward(grad.data, t1, t2)

        return AddBackward

class Mul(TensorBinaryFunction):
    from synapse.autograd.tensor import Tensor, Node

    def function(self, t1: Tensor, t2: Tensor) -> Tensor:
        from synapse.autograd.tensor import Tensor
        if not t1.shape == t2.shape:
            raise RuntimeError("Broadcasting not Implemented (yet)")
        else:
            data = t1.data * t2.data

        requiresGrad = t1.requiresGrad or t2.requiresGrad
        resultTensor = Tensor(data, requiresGrad)

        return resultTensor


    def gradFn0(self, t1: Tensor, t2: Tensor) -> Tensor:
        from synapse.autograd.gradfns import mulBackward0
        from synapse.autograd.tensor import Tensor

        def MulBackward0(grad: Tensor) -> Callable[[np.ndarray, Tensor, Tensor], Tensor]:
            return mulBackward0(grad.data, t1, t2)

        return MulBackward0

    def gradFn1(self, t1: Tensor, t2: Tensor) -> Tensor:
        from synapse.autograd.gradfns import mulBackward1
        from synapse.autograd.tensor import Tensor

        def MulBackward1(grad: Tensor) -> Callable[[np.ndarray, Tensor, Tensor], Tensor]:
            return mulBackward1(grad.data, t1, t2)
        return MulBackward1

class MatMul(TensorBinaryFunction):
    from synapse.autograd.tensor import Tensor, Node

    def function(self, t1: Tensor, t2: Tensor) -> Tensor:
        from synapse.autograd.tensor import Tensor
        data = np.matmul(t1.data, t2.data)

        requiresGrad = t1.requiresGrad or t2.requiresGrad
        resultTensor = Tensor(data, requiresGrad)

        return resultTensor


    def gradFn0(self, t1: Tensor, t2: Tensor) -> Tensor:
        from synapse.autograd.gradfns import matmulBackward0
        from synapse.autograd.tensor import Tensor

        def MatMulBackward0(grad: Tensor) -> Callable[[np.ndarray, Tensor, Tensor], Tensor]:
            return matmulBackward0(grad.data, t1, t2)

        return MatMulBackward0

    def gradFn1(self, t1: Tensor, t2: Tensor) -> Tensor:
        from synapse.autograd.gradfns import matmulBackward1
        from synapse.autograd.tensor import Tensor

        def MatMulBackward1(grad: Tensor) -> Callable[[np.ndarray, Tensor, Tensor], Tensor]:
            return matmulBackward1(grad.data, t1, t2)

        return MatMulBackward1

tensorSum = Sum()
tensorAdd = Add()
tensorMul = Mul()
matmul = MatMul()
