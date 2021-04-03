import pytest
from unittest import TestCase

from sparknet.nn.model import Model
from sparknet.nn.layers import Linear
from sparknet.autograd.tensor import Tensor

import numpy as np

class TestNN(TestCase):
    def testModel(self):

        class NN(Model):
            def __init__(self):
                self.linear1 = Linear(64, 32)
                self.linear2 = Linear(32, 16)
                self.linear3 = Linear(16, 8)
                self.linear4 = Linear(16, 4)
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
        testInput = Tensor(np.random.randn(64, 32))
        testOutput = model(testInput)
        assert testOutput.shape == (2, 1)



