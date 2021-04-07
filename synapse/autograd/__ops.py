
from typing import Callable, Union
import numpy as np
from .__tensorFunctions import TensorFunction, TensorBinaryFunction

Number = Union(int, float)
GradFn = Callable[[Tensor], Tensor]

class Sum(TensorFunction):
    from synapse.autograd.tensor import Tensor, Node

    def forward(self, t1: Tensor) -> Tensor:
        from synapse.autograd.tensor import Tensor
        data = np.sum(t1.data)
        requiresGrad = t1.requiresGrad
        resultTensor = Tensor(data, requiresGrad)

        return resultTensor


    def gradFn(self, t1: Tensor) -> GradFn:
        from synapse.autograd.__gradFns import sumBackward
        from synapse.autograd.tensor import Tensor

        def SumBackward(grad: Tensor) -> Tensor:
            return sumBackward(grad.data, t1)

        return SumBackward

class Exponentiate(TensorFunction):
    from synapse.autograd.tensor import Tensor

    def __call__(self, t1: Tensor, power: Number) -> Tensor:
        resultTensor = self.forward(t1, power)
        if t1.requiresGrad:
            resultTensor.addParent(self.gradFn(t1, power))
        return resultTensor

    def forward(self, t1: Tensor, power: Number) -> Tensor:
        from synapse.autograd.tensor import Tensor
        data = t1.data ** power
        requiresGrad = t1.requiresGrad
        resultTensor = Tensor(data, requiresGrad)

        return resultTensor


    def gradFn(self, t1: Tensor, power: Number) -> GradFn:
        from synapse.autograd.__gradFns import sumBackward
        from synapse.autograd.tensor import Tensor

        def ExpBackward(grad: Tensor) -> Tensor:
            data = grad.data * power * (t1.data ** (power-1) )
            requiresGrad = t1.requiresGrad
            return Tensor(data, requiresGrad)

        return SumBackward

class Mean(TensorFunction):
    from synapse.autograd.tensor import Tensor

    def forward(self, t1: Tensor) -> Tensor:
        from synapse.autograd.tensor import Tensor
        data = np.mean(t1.data)
        requiresGrad = t1.requiresGrad
        resultTensor = Tensor(data, requiresGrad)

        return resultTensor


    def gradFn(self, t1: Tensor) -> GradFn:
        from synapse.autograd.__gradFns import meanBackward
        from synapse.autograd.tensor import Tensor

        def MeanBackward(grad: Tensor) -> Tensor:
            data = np.ones_like(t1.data) / np.size(t1.data)
            data = grad * data
            return Tensor(data)

        return MeanBackward

class Neg(TensorFunction):
    from synapse.autograd.tensor import Tensor, Node

    def forward(self, t1: Tensor) -> Tensor:
        from synapse.autograd.tensor import Tensor
        data = - t1.data
        requiresGrad = t1.requiresGrad
        resultTensor = Tensor(data, requiresGrad)
        return resultTensor


    def gradFn(self, t1: Tensor) -> GradFn:
        from synapse.autograd.__gradFns import sumBackward
        from synapse.autograd.tensor import Tensor

        def negBackward(grad: Tensor) -> Tensor:

            data = np.negative(grad.data)
            return Tensor(data, t1.requiresGrad)

        return negBackward

class Add(TensorBinaryFunction):
    from synapse.autograd.tensor import Tensor, Node

    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        from synapse.autograd.tensor import Tensor
        if not t1.shape == t2.shape:
            raise RuntimeError("Broadcasting not Implemented (yet)")
        else:
            data = t1.data + t2.data

        requiresGrad = t1.requiresGrad or t2.requiresGrad
        resultTensor = Tensor(data, requiresGrad)

        return resultTensor


    def gradFn0(self, t1: Tensor, t2: Tensor) -> GradFn:
        from synapse.autograd.__gradFns import addBackward
        from synapse.autograd.tensor import Tensor

        def AddBackward(grad: Tensor) -> Tensor:
            return addBackward(grad.data, t1, t2)

        return AddBackward

    def gradFn1(self, t1: Tensor, t2: Tensor) -> GradFn:
        from synapse.autograd.__gradFns import addBackward
        from synapse.autograd.tensor import Tensor

        def AddBackward(grad: Tensor) -> Tensor:
            return addBackward(grad.data, t1, t2)

        return AddBackward

class Mul(TensorBinaryFunction):
    from synapse.autograd.tensor import Tensor, Node

    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        from synapse.autograd.tensor import Tensor
        if not t1.shape == t2.shape:
            raise RuntimeError("Broadcasting not Implemented (yet)")
        else:
            data = t1.data * t2.data

        requiresGrad = t1.requiresGrad or t2.requiresGrad
        resultTensor = Tensor(data, requiresGrad)

        return resultTensor


    def gradFn0(self, t1: Tensor, t2: Tensor) -> GradFn:
        from synapse.autograd.__gradFns import mulBackward0
        from synapse.autograd.tensor import Tensor

        def MulBackward0(grad: Tensor) -> Tensor:
            return mulBackward0(grad.data, t1, t2)

        return MulBackward0

    def gradFn1(self, t1: Tensor, t2: Tensor) -> GradFn:
        from synapse.autograd.__gradFns import mulBackward1
        from synapse.autograd.tensor import Tensor

        def MulBackward1(grad: Tensor) -> Tensor:
            return mulBackward1(grad.data, t1, t2)
        return MulBackward1

class MatMul(TensorBinaryFunction):
    from synapse.autograd.tensor import Tensor, Node

    def forward(self, t1: Tensor, t2: Tensor) -> Tensor:
        from synapse.autograd.tensor import Tensor
        data = np.matmul(t1.data, t2.data)

        requiresGrad = t1.requiresGrad or t2.requiresGrad
        resultTensor = Tensor(data, requiresGrad)

        return resultTensor


    def gradFn0(self, t1: Tensor, t2: Tensor) -> GradFn:
        from synapse.autograd.__gradFns import matmulBackward0
        from synapse.autograd.tensor import Tensor

        def MatMulBackward0(grad: Tensor) -> Tensor:
            return matmulBackward0(grad.data, t1, t2)

        return MatMulBackward0

    def gradFn1(self, t1: Tensor, t2: Tensor) -> GradFn:
        from synapse.autograd.__gradFns import matmulBackward1
        from synapse.autograd.tensor import Tensor

        def MatMulBackward1(grad: Tensor) -> Tensor:
            return matmulBackward1(grad.data, t1, t2)

        return MatMulBackward1

tensorSum = Sum()
tensorAdd = Add()
tensorMul = Mul()
tensorNeg = Neg()
tensorMean = Mean()
matmul = MatMul()
tensorExp = Exponentiate()
