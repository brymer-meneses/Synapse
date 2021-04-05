import pytest
from unittest import TestCase

from synapse import Tensor
from synapse.nn.activations import Tanh, ReLU

import numpy as np
from numpy.testing import assert_array_equal

class TestActivations(TestCase):


    def testTanh(self):
        tanh = Tanh()
        testData = Tensor(np.random.uniform(-10, 10, size=(5,5)), requiresGrad =True)
        y = tanh(testData)

        initialGrad = Tensor(np.ones_like(testData.data))
        y.backward(initialGrad)

        assert_array_equal(y.data, np.tanh(testData.data))
        assert_array_equal(testData.grad.data, 1 - np.tanh(testData.data)**2)

        return
    def testReLU(self):
        relu = ReLU()
        testData = Tensor(np.random.uniform(-10, 10, size=(5,5)), requiresGrad=True)
        correctResult = np.where(testData.data > 0, testData.data, 0)
        y = relu(testData)

        initialGrad = Tensor(np.ones_like(testData.data))
        y.backward(initialGrad)

        assert_array_equal(y.data, correctResult)
        assert_array_equal(testData.grad.data, np.where(testData.data<0, 0, 1))
