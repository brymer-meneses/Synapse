import sys
sys.path.append("../synapse")

import synapse as sn
from synapse import Tensor
from synapse.nn.model import Model
from synapse.nn.layers import Linear
from synapse.nn.loss import MSE
from synapse.nn.optimizers import SGD
from synapse.nn.activations import ReLU, Tanh

from typing import List
import numpy as np
np.random.seed(10)


class NeuralNet(Model):
    def __init__(self):
        self.linear1 = Linear(1, 64)
        self.linear2 = Linear(64, 32)
        self.linear3 = Linear(32, 16)
        self.linear4 = Linear(16, 8)
        self.linear5 = Linear(8, 4)
        self.tanh = Tanh()
    def forward(self, x):
        x = self.tanh(self.linear1(x))
        x = self.tanh(self.linear2(x))
        x = self.tanh(self.linear3(x))
        x = self.tanh(self.linear4(x))
        x = self.tanh(self.linear5(x))
        return x

def generateData(nI: int, nF: int) -> List[Tensor]:
    inputs = []
    targets = []
    for num in range(nI, nF):
        inputs.append(num)
        if num % 15:
            targets.append([0, 0, 1, 0])
            continue
        elif num % 3:
            targets.append([1, 0, 0, 0])
            continue
        elif num % 5:
            targets.append([0, 1, 0, 0])
            continue
        else:
            targets.append([0, 0, 0, 1])
            continue
    targets = np.array(targets).T
    targetTensor = Tensor(targets, requiresGrad=False)
    inputs = np.expand_dims(np.array(inputs), axis=0)
    inputTensor = Tensor(inputs, requiresGrad=False)
    return inputTensor, targetTensor



model = NeuralNet()
sgd = SGD(lr=-0.1)
mse = MSE()
model.compile(sgd, mse)
model.summary()

x_train, y_train = generateData(101, 501)


for epoch in range(100):
    output = model(x_train)

    loss = mse(output, y_train)
    loss.backward()
    model.optimize()

print(model(Tensor([[15.0]])))



