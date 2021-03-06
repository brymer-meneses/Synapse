import pytest
from unittest import TestCase

from synapse import Tensor
from synapse.nn.activations import Tanh, ReLU
from synapse.nn.loss import MSE
from synapse.core.differentiable import Differentiable

import numpy as np
from numpy.testing import assert_array_equal
from synapse.debugging import show_parents

class TestActivations(TestCase):


    def testTanh(self):
        testData = Tensor(np.random.uniform(-10, 10, size=(5,5)), requires_grad =True)
        y = Tanh(testData)

        initialGrad = Tensor(np.ones_like(testData.data))
        y.backward(initialGrad)

        assert_array_equal(y.data, np.tanh(testData.data))
        assert_array_equal(testData.grad.data, 1 - np.tanh(testData.data)**2)

        return
    def testReLU(self):
        testData = Tensor(np.random.uniform(-10, 10, size=(5,5)), requires_grad=True)
        correctResult = np.where(testData.data > 0, testData.data, 0)
        y = ReLU(testData)

        initialGrad = Tensor(np.ones_like(testData.data))
        y.backward(initialGrad)

        assert_array_equal(y.data, correctResult)
        assert_array_equal(testData.grad.data, np.where(testData.data>0, 1, 0))
