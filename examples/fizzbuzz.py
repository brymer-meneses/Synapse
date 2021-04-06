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


class NeuralNet(Model):
    def __init__(self):
        self.linear1 = Linear(32, 16)
        self.relu1 = ReLU()
        self.linear2 = Linear(16, 8)
        self.relu2 = ReLU()
        self.linear3 = Linear(8, 4)
        self.tanh = Tanh()
    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.tanh(self.linear3(x))
        return x

def generateData(nI: int, nF: int) -> List[Tensor]:
    inputs = []
    targets = []
    for num in range(nI, nF):
        inputs.append(num)
        if num % 3:
            targets.append([1, 0, 0, 0])
        if num % 5:
            targets.append([0, 1, 0, 0])
        if num % 15:
            targets.append([0, 0, 1, 0])
        else:
            targets.append([0, 0, 0, 1])
    targets = np.array(targets)
    targetTensor = Tensor(targets, requiresGrad=False)
    inputs = np.array(inputs)
    inputTensor = Tensor(inputs, requiresGrad=False)
    return targetTensor, inputTensor



model = NeuralNet()
sgd = SGD()
mse = MSE()
model.compile(sgd, mse)
model.summary()

x_train, y_train = generateData(101, 500)
x_train.data, y_train.data = x_train.data[0:32], y_train.data[0:32]
print(x_train.shape)
model.fit(x_train, y_train, 10)



