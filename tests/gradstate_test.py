

from unittest import TestCase
import synapse as sn
import pytest
import numpy as np

class TestGradMode(TestCase):
    def testNoGrad(self):
        data1 = np.random.uniform(-10, 10, size=(5,5))
        data2 = np.random.uniform(-10, 10, size=(5,5))
        with sn.no_grad():


            a = sn.Tensor(data1, requires_grad=True)
            b = sn.Tensor(data2, requires_grad=True)
            c = a.sum()
            d = a + b
            e = sn.matmul(a, b)
            f = a * b

            assert a.requires_grad == False
            assert b.requires_grad == False
            assert c.requires_grad == False
            assert d.requires_grad == False
            assert e.requires_grad == False
            assert f.requires_grad == False
