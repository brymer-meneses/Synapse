import sys
sys.path.append("../synapse")

from synapse import Tensor
from synapse.nn.model import Model
from synapse.nn.layers import Linear
from synapse.nn.optimizers import SGD

import numpy as np

class NeuralNet(Model):
    def __init__(self):
        self.linear1 = Linear(10, 5)
        self.linear2 = Linear(5, 1)
        self.linear3 = Linear(1,1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

sgd = SGD(lr=0.002)
model = NeuralNet()
model.compile(sgd)
model.summary()

testData = Tensor(np.random.uniform(-10, 10, size=(10, 1)))
output = model(testData)

for node in output.parentNodes:
    print(node)

print(model.linear1.weights.grad)
print(model.linear2.weights.grad)
print(model.linear3.weights.grad)

output.backwards(Tensor([[1.0]]))

print(model.linear1.weights.grad)
print(model.linear2.weights.grad)
print(model.linear3.weights.grad)



