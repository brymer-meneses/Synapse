
from unittest import TestCase

import sparknet as sn
import numpy as np
from numpy.testing import assert_array_equal

class TestAutograd(TestCase):
    def testMulGrad(self):
        d1 = np.random.uniform(0,10, size=(5,5))
        d2 = np.random.uniform(0,10, size=(5,5))

        t1 = sn.tensor(d1, requiresGrad=True)
        t2 = sn.tensor(d2, requiresGrad=True)
        t3 = t1 * t2

        initialGrad = sn.tensor(np.ones_like(t1.data))
        t3.backwards(initialGrad)

        assert_array_equal(t1.grad.data, t2.data)
        assert_array_equal(t2.grad.data, t1.data)

    def testAddGrad(self):
        d1 = np.random.uniform(0,10, size=(5,5))
        d2 = np.random.uniform(0,10, size=(5,5))

        t1 = sn.tensor(d1, requiresGrad=True)
        t2 = sn.tensor(d2, requiresGrad=True)
        t3 = t1 + t2

        initialGrad = sn.tensor(np.ones_like(t1.data))
        t3.backwards(initialGrad)

        assert_array_equal(t1.grad.data, np.ones_like(t2.data))
        assert_array_equal(t2.grad.data, np.ones_like(t1.data))

    def testSumGrad(self):
        d1 = np.random.uniform(0,10, size=(5,5))
        t1 = sn.tensor(d1, requiresGrad=True)

        t2 = t1.sum()
        t2.backwards()

        assert_array_equal(t2.grad, np.ones_like(t1.data))
        assert t2.requiresGrad == True

    def testMatMulGrad(self):
        d1 = np.random.uniform(0,10, size=(5,5))
        d2 = np.random.uniform(0,10, size=(5,5))

        t1 = sn.tensor(d1, requiresGrad=True)
        t2 = sn.tensor(d2, requiresGrad=True)

        t3 = sn.matmul(t1, t2)

        initialGrad = sn.tensor(np.ones_like(t3.data))
        t3.backwards(initialGrad)

        assert_array_equal(t1.grad.data, np.matmul(initialGrad, t2.T))
        assert_array_equal(t2.grad.data, np.matmul(t1.data.T, initialGrad))

