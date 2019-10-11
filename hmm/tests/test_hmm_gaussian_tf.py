import os
import logging
import unittest
import numpy as np
import tensorflow as tf
from ..hmm_gaussian_tf import HMMGaussianTF
from . import utils

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel(logging.ERROR)


class TestHMMGaussianTF(unittest.TestCase):

    NUM_HIDDEN_STATES = 3
    DIMENSIONALITY = 7
    SEQ_LENGTH = 11

    def tearDown(self) -> None:
        tf.reset_default_graph()

    def test_forward(self):

        model = HMMGaussianTF(self.NUM_HIDDEN_STATES, self.DIMENSIONALITY, self.SEQ_LENGTH)
        model.setup_variables()
        model.setup_placeholders()

        alphas, log_evidence = model.forward()

        self.assertEqual(utils.get_tensor_shape(alphas), (None, self.SEQ_LENGTH, self.NUM_HIDDEN_STATES))
        self.assertEqual(utils.get_tensor_shape(log_evidence), (None,))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            alphas, log_evidence = sess.run([alphas, log_evidence], feed_dict={
                model.seq: np.random.uniform(-1, 1, size=(13, 11, 7))
            })
            self.assertTrue(np.all(np.bitwise_not(np.isnan(alphas))))
            self.assertTrue(np.all(np.bitwise_not(np.isnan(log_evidence))))

    def test_backward(self):

        model = HMMGaussianTF(self.NUM_HIDDEN_STATES, self.DIMENSIONALITY, self.SEQ_LENGTH)
        model.setup_variables()
        model.setup_placeholders()

        log_betas = model.backward()
        self.assertEqual(utils.get_tensor_shape(log_betas), (None, self.SEQ_LENGTH, self.NUM_HIDDEN_STATES))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            log_betas_val = sess.run(log_betas, feed_dict={
                model.seq: np.random.uniform(-1, 1, size=(13, 11, 7))
            })
            self.assertTrue(np.all(np.bitwise_not(np.isnan(log_betas_val))))

    def test_forward_backward(self):

        model = HMMGaussianTF(self.NUM_HIDDEN_STATES, self.DIMENSIONALITY, self.SEQ_LENGTH)
        model.setup_variables()
        model.setup_placeholders()

        alphas, log_evidence, betas, gammas, etas = model.forward_backward()

        self.assertEqual(utils.get_tensor_shape(alphas), (None, self.SEQ_LENGTH, self.NUM_HIDDEN_STATES))
        self.assertEqual(utils.get_tensor_shape(log_evidence), (None,))
        self.assertEqual(utils.get_tensor_shape(betas), (None, self.SEQ_LENGTH, self.NUM_HIDDEN_STATES))
        self.assertEqual(utils.get_tensor_shape(gammas), (None, self.SEQ_LENGTH, self.NUM_HIDDEN_STATES))
        self.assertEqual(utils.get_tensor_shape(etas),
                         (None, self.SEQ_LENGTH - 1, self.NUM_HIDDEN_STATES, self.NUM_HIDDEN_STATES))

    def test_likelihood(self):

        model = HMMGaussianTF(self.NUM_HIDDEN_STATES, self.DIMENSIONALITY, self.SEQ_LENGTH)
        model.setup_variables()
        model.setup_placeholders()

        log_likelihood = model.likelihood()
        self.assertEqual(len(log_likelihood.shape), 0)
