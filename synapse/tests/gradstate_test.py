

from unittest import TestCase
import synapse as sn
import pytest
import numpy as np

class TestGradMode(TestCase):
    def testNoGrad(self):
        data1 = np.random.uniform(-10, 10, size=(5,5))
        data2 = np.random.uniform(-10, 10, size=(5,5))
        with sn.NoGrad():


            a = sn.Tensor(data1, requiresGrad=True)
            b = sn.Tensor(data2, requiresGrad=True)
            c = a.sum()
            d = a + b
            e = sn.matmul(a, b)
            f = a * b

            assert a.requiresGrad == False
            assert b.requiresGrad == False
            assert c.requiresGrad == False
            assert d.requiresGrad == False
            assert e.requiresGrad == False
            assert f.requiresGrad == False
