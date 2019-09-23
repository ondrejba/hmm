import unittest
import numpy as np
from ..casino import Casino
from ..hmm import HMMMultinoulli


class TestHMMMultinoulli(unittest.TestCase):

    def setUp(self):
        np.random.seed(2019)

    def test_condition(self):

        hmm = HMMMultinoulli(Casino.A, Casino.PX, Casino.INIT)
        seq = [2, 4, 3, 5, 1, 0, 3, 2, 1, 5, 4]
        output = hmm.condition(seq)

        self.assertEqual(output.shape, (2, len(seq)))

        for i, s in enumerate(seq):
            np.testing.assert_equal(output[:, i], Casino.PX[:, s])
