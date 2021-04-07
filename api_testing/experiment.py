import sys
sys.path.append("../synapse")

from synapse import Tensor
from synapse.nn.model import Model
from synapse.nn.layers import Linear
from synapse.nn.optimizers import SGD
from synapse.nn.loss import MSE
from synapse.nn.activations import ReLU, Tanh

import numpy as np
np.random.seed(10)
class NeuralNet(Model):
    def __init__(self):
        self.linear1 = Linear(10, 5)
        self.linear2 = Linear(5, 1)
        self.tanh = Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(self.linear2(x))
        return self.tanh(x)

sgd = SGD(lr=0.002)
model = NeuralNet()
model.compile(sgd, MSE())
model.summary()

testData = Tensor(np.random.uniform(-10, 10, size=(10, 1)))
output = model(testData)
print(model.linear2.weights.grad)
print(model.linear1.weights.grad)

output.backward(Tensor([[1.0]]))
print("\n")
print("no optimize")
# model.optimize()
print(model.linear2.weights.grad)
print(model.linear1.weights.grad)



