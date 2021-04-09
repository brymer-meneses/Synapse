
import sys
sys.path.append("../synapse")

import numpy as np
import synapse as sn
from synapse import Tensor
from synapse.autograd.tensor import Node
from synapse.nn.loss import MSE
from synapse.nn.activations import ReLU
from synapse.testing.graph import showParents

data = np.arange(1,10, dtype=np.float)
data = np.expand_dims(data, 0)

x = Tensor(data)
print(x.shape)

W = Tensor(np.zeros_like(data),requiresGrad=True)
y = Tensor(2*data)

 
maxEpoch = 1000
lr = 0.1
initialGrad = sn.Tensor([[1.0]])

for epoch in range(maxEpoch):
    predicted = W *x

    loss = MSE(predicted, y)
    loss.backward()
    W.data = W.data - lr * W.grad.data
    W.zeroGrad()
    print(loss.data)

print(f"Prediction: {W.data}")


