import pytest
from unittest import TestCase

from synapse import Tensor
from synapse.nn.activations import Tanh, ReLU

import numpy as np
from numpy.testing import assert_array_equal

class TestActivations(TestCase):


    def testTanh(self):
        tanh = Tanh()
        testData = Tensor(np.random.uniform(-10, 10, size=(5,5)))
        y = tanh(testData)
        print("test")

        assert_array_equal(y.data, np.tanh(testData.data))

        return
    def testReLU(self):
        relu = ReLU()
        testData = Tensor(np.random.uniform(-10, 10, size=(5,5)))
        correctResult = np.where(testData.data > 0, testData.data, 0)
        y = relu(testData)

        assert_array_equal(y.data, correctResult)

