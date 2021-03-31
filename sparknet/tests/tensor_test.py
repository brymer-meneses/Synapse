
from unittest import TestCase
import pytest

from sparknet import tensor
import numpy as np

class TestTensorOps(TestCase):
    def test_add(self):
        data1 = np.random.randn(5,5)
        data2 = np.random.randn(5.5)

        t1 = tensor(data1, requiresGrad=True)
        t2 = tensor(data2, requiresGrad=True)
        t3 = t1 + t2
        assert t3.data.tolist() == data1 + data2
        assert t3.requiresGrad == True
        assert t1.requiresGrad == True
        assert t2.requiresGrad == True

        return
    def test_mul(self):
        data1 = np.random.randn(5,5)
        data2 = np.random.randn(5,5)

        t1 = tensor(data1, requiresGrad=True)
        t2 = tensor(data2, requiresGrad=True)
        t3 = t1 * t2


