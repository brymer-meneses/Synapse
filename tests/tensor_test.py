
from unittest import TestCase
import pytest

import synapse as sn
from synapse import Tensor
import numpy as np
from numpy.testing import assert_array_equal

class TestTensorOps(TestCase):

    def testSum(self):
        data1 = np.random.uniform(0, 10, size=(5,5))

        t1 = Tensor(data1, requires_grad=True)
        t2 = t1.sum()

        assert_array_equal(t2.data, data1.sum())
        assert t2.requires_grad == True

    def testMean(self):
        data1 = np.random.uniform(0, 10, size=(5,5))
        t1 = Tensor(data1, requires_grad=True)
        t2 = t1.mean()
        t2.backward()
        assert_array_equal(t1.grad.data, np.ones_like(data1) / np.size(data1))

    def testPow(self):
        data1 = np.random.uniform(0, 10, size=(5,5))
        initialGrad = Tensor(np.ones_like(data1))
        t1 = Tensor(data1, requires_grad=True)
        t2 = sn.pow(t1, 5)
        t2.backward(initialGrad)
        assert_array_equal(t1.grad.data,5 * data1 ** (4))

    def testNeg(self):
        data1 = np.random.uniform(0, 10, size=(5,5))
        t1 = Tensor(data1, requires_grad=True)
        t2 = - t1
        initialGrad = Tensor(np.ones_like(data1))
        t2.backward(initialGrad)
        assert_array_equal(t1.grad.data,- initialGrad.data)

    def testAdd(self):
        data1 = np.random.uniform(0, 10, size=(5,5))
        data2 = np.random.uniform(0, 10, size=(5,5))

        t1 = Tensor(data1, requires_grad=True)
        t2 = Tensor(data2, requires_grad=True)
        t3 = t1 + t2

        assert_array_equal(t3.data, data1 + data2)
        assert t3.requires_grad == True

        return
    def testMul(self):
        data1 = np.random.uniform(0, 10, size=(5,5))
        data2 = np.random.uniform(0, 10, size=(5,5))

        t1 = Tensor(data1, requires_grad=True)
        t2 = Tensor(data2, requires_grad=True)
        t3 = t1 * t2

        assert_array_equal(t3.data, data1 * data2)
        assert t3.requires_grad == True

    def testMatMul(self):
        data1 = np.random.uniform(0, 10, size=(5,5))
        data2 = np.random.uniform(0, 10, size=(5,5))

        t1 = Tensor(data1, requires_grad=True)
        t2 = Tensor(data2, requires_grad=True)
        t3 = sn.matmul(t1, t2)

        assert_array_equal(t3.data, np.matmul(data1, data2))
        assert t3.requires_grad == True



