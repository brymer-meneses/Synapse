
from unittest import TestCase
import pytest

import sparknet as sn
from sparknet import tensor
import numpy as np

class TestTensorOps(TestCase):
    def testSum(self):
        data1 = np.random.uniform(0, 10, size=(5,5))

        t1 = tensor(data1, requiresGrad=True)
        t2 = t1.sum()

        assert t2.data.tolist() == t1.data.sum()
        assert t2.requiresGrad == True

    def testAdd(self):
        data1 = np.random.uniform(0, 10, size=(5,5))
        data2 = np.random.uniform(0, 10, size=(5,5))

        t1 = tensor(data1, requiresGrad=True)
        t2 = tensor(data2, requiresGrad=True)
        t3 = t1 + t2

        assert t3.data.tolist() == (data1 + data2).tolist()
        assert t3.requiresGrad == True

        return
    def testMul(self):
        data1 = np.random.uniform(0, 10, size=(5,5))
        data2 = np.random.uniform(0, 10, size=(5,5))

        t1 = tensor(data1, requiresGrad=True)
        t2 = tensor(data2, requiresGrad=True)
        t3 = t1 * t2

        assert t3.data.tolist() == (data1 * data2).tolist()
        assert t3.requiresGrad == True

    def testMatmul(self):
        data1 = np.random.uniform(0, 10, size=(5,5))
        data2 = np.random.uniform(0, 10, size=(5,5))

        t1 = tensor(data1, requiresGrad=True)
        t2 = tensor(data2, requiresGrad=True)
        t3 = sn.matmul(t1, t2)

        assert t3.data.tolist() == np.matmul(data1, data2).tolist()
        assert t3.requiresGrad == True



