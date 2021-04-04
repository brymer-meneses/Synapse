import pytest
from unittest import TestCase

from sparknet import tensor
from sparknet.nn.activations import Tanh, ReLU

import numpy as np
from numpy.testing import assert_array_equal

class TestActivations(TestCase):
    def testTanh(self):
        tanh = Tanh()
        testData = tensor(np.random.uniform(-10, 10, size=(5,5)))
        y = tanh(testData)

        assert_array_equal(y.data, np.tanh(testData.data))

        return
    def testReLU(self):
        relu = ReLU()
        testData = tensor(np.random.uniform(-10, 10, size=(5,5)))
        correctResult = np.where(testData.data > 0, testData.data, 0)
        y = relu(testData)

        assert_array_equal(y.data, correctResult)
