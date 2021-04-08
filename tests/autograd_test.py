
from unittest import TestCase

import synapse as sn
import numpy as np
from numpy.testing import assert_array_equal
from synapse.testing.graph import showParents

class TestAutograd(TestCase):
    def testMulGrad(self):
        d1 = np.random.uniform(0,10, size=(5,5))
        d2 = np.random.uniform(0,10, size=(5,5))

        t1 = sn.Tensor(d1, requiresGrad=True)
        t2 = sn.Tensor(d2, requiresGrad=True)
        t3 = t1 * t2 # 5x5

        initialGrad = sn.Tensor(np.ones_like(t1.data))
        t3.backward(initialGrad)

        assert_array_equal(t1.grad.data, t2.data)
        assert_array_equal(t2.grad.data, t1.data)

    def testAddGrad(self):
        d1 = np.random.uniform(0,10, size=(5,5))
        d2 = np.random.uniform(0,10, size=(5,5))

        t1 = sn.Tensor(d1, requiresGrad=True)
        t2 = sn.Tensor(d2, requiresGrad=True)
        t3 = t1 + t2

        initialGrad = sn.Tensor(np.ones_like(t1.data))
        t3.backward(initialGrad)

        assert_array_equal(t1.grad.data, np.ones_like(t2.data))
        assert_array_equal(t2.grad.data, np.ones_like(t1.data))

    def testSumGrad(self):
        d1 = np.random.uniform(0,10, size=(5,5))
        t1 = sn.Tensor(d1, requiresGrad=True)

        t2 = t1.sum()
        t2.backward()

        assert_array_equal(t2.grad.data, np.ones_like(t1.data))
        assert t2.requiresGrad == True

    def testMatMulGrad(self):
        d1 = np.random.uniform(0,10, size=(5,5))
        d2 = np.random.uniform(0,10, size=(5,5))

        t1 = sn.Tensor(d1, requiresGrad=True)
        t2 = sn.Tensor(d2, requiresGrad=True)

        t3 = sn.matmul(t1, t2) # 5x5

        initialGrad = sn.Tensor(np.ones_like(t3.data))
        t3.backward(initialGrad)

        assert_array_equal(t1.grad.data, np.matmul(initialGrad.data, t2.data.T))
        assert_array_equal(t2.grad.data, np.matmul(t1.data.T, initialGrad.data))

