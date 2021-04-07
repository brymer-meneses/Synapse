
import sys
sys.path.append("../synapse")

import synapse.nn as nn
from synapse.nn import layers,activations, loss
from synapse import Tensor

import numpy as np

class NeuralNet(nn.Model):
    def __init__(self):
        self.linear1 = layers.Linear(10, 5)
        self.relu1 = activations.ReLU()
        self.linear2 = layers.Linear(5, 1)
        self.relu2 = activations.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return x

model = NeuralNet()
testData = Tensor(np.random.uniform(-1, 1, (10,10)))
output = model(testData)
loss = loss.MSE()

output.backward(Tensor([[1.0]]))

