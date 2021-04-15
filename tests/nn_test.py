import pytest
from unittest import TestCase

from synapse.nn.model import Model
from synapse.nn.layers import Linear
from synapse.nn.optimizers import SGD
from synapse.nn.loss import MSE

from synapse.core.tensor import Tensor

import numpy as np

class TestNN(TestCase):
    def testModelBackward(self):
        class NN(Model):
            def __init__(self):
                self.linear1 = Linear(10, 5)
                self.linear2 = Linear(5, 1)
            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        model = NN()
        sgd = SGD(lr=0.01)
        model.compile(sgd, MSE)

        testData = Tensor(np.random.uniform(10, -10, size=(10,1)))
        output = model(testData)
        testGrad = Tensor(np.random.uniform(10, -10, size=(1,1)))
        output.backward(testGrad)
        return
    def testModelCreation(self):

        class NN(Model):
            def __init__(self):
                self.linear1 = Linear(64, 32)
                self.linear2 = Linear(32, 16)
                self.linear3 = Linear(16, 8)
                self.linear4 = Linear(8, 4)
                self.linear5 = Linear(4, 2)
                self.final = Linear(2, 1)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.linear3(x)
                x = self.linear4(x)
                x = self.linear5(x)
                x = self.final(x)
                return x

        model = NN()
        sgd = SGD(lr=0.002)
        model.compile(sgd, MSE )
        # model.summary()

        testInput = Tensor(np.random.randn(64, 1))
        testOutput = model(testInput)
        assert testOutput.shape == (1, 1)





