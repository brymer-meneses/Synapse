from synapse.autograd import TensorFunction
from synapse import Tensor


class Softmax(TensorFunction):
    def function(self, t1: Tensor) -> Tensor:
        expData = np.exp(t1.data)
        data = expData / np.sum(expData, axis=0)
        requiresGrad = t1.requiresGrad
        return Tensor(expData, requiresGrad)

    def gradFn0(self, t1: Tensor) -> Tensor:
        def SoftmaxBackward(grad: np.ndarray) -> Callable[[np.ndarray], Tensor]

