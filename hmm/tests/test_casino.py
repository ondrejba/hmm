import unittest
import numpy as np
from ..casino import Casino


class TestCasino(unittest.TestCase):

    def setUp(self):
        np.random.seed(2019)

    def test_transition(self):

        c = Casino()
        zs = []

        for i in range(10000):

            c.transition()
            zs.append(c.z)

        print(np.unique(zs, return_counts=True))

    def test_observe(self):

        c1 = Casino()
        c2 = Casino()

        c1.z = Casino.Z_HONEST
        c2.z = Casino.Z_DISHONEST

        o1, o2 = [], []

        for i in range(10000):
            o1.append(c1.observe())
            o2.append(c2.observe())

        print(np.unique(o1, return_counts=True))
        print(np.unique(o2, return_counts=True))
