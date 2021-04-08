import pytest
from unittest import TestCase

import synapse as sn
import numpy as np
from numpy.testing import assert_array_equal
from synapse.nn.loss import MSE
from synapse.testing.graph import showParents

class TestLoss(TestCase):
    def testMSE(self):
        mse = MSE()
        data1 = np.random.uniform(-10, 10, size=(3,3))
        data2 = np.random.uniform(-10, 10, size=(3,3))
        a = sn.Tensor(data1, True)
        b = sn.Tensor(data2, False)
        c = mse(a, b)
        showParents(c)
        c.backward()

        realGrad = np.multiply(2, (data1 - data2).mean())


        assert_array_equal(a.grad.data, realGrad)
