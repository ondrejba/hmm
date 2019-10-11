import os
import logging
import unittest
import numpy as np
import tensorflow as tf
from ..hmm_gaussian import HMMGaussian
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

    def diag_batch(self, x):

        y = []

        for batch_idx in range(x.shape[0]):
            y.append(np.diag(x[batch_idx]))

        return np.stack(y, axis=0)

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

    def test_condition_match_ref(self):

        seq_batch = np.random.uniform(-1, 1, size=(13, self.SEQ_LENGTH, self.DIMENSIONALITY))
        seq = seq_batch[0]

        model = HMMGaussianTF(self.NUM_HIDDEN_STATES, self.DIMENSIONALITY, self.SEQ_LENGTH)
        model.setup()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            A, init, mu, cov = sess.run([model.A, model.init, model.mu, model.cov])
            full_cov = self.diag_batch(cov)

            model_np = HMMGaussian(A, init, mu, full_cov)

            log_px_ref = model_np.condition(seq)
            log_px_ref = np.transpose(log_px_ref)
            log_px = sess.run(model.log_condition_seq, feed_dict={model.seq: seq_batch})[0]

            np.testing.assert_almost_equal(log_px, log_px_ref, decimal=2)

    def test_match_ref(self):

        seq_batch = np.random.uniform(-1, 1, size=(13, self.SEQ_LENGTH, self.DIMENSIONALITY))
        seq = seq_batch[0]

        model = HMMGaussianTF(self.NUM_HIDDEN_STATES, self.DIMENSIONALITY, self.SEQ_LENGTH)
        model.setup()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            A, init, mu, cov = sess.run([model.A, model.init, model.mu, model.cov])
            full_cov = self.diag_batch(cov)

            #print(cov.shape, full_cov.shape)

            model_np = HMMGaussian(A, init, mu, full_cov)

            log_alphas_ref, log_evidence_ref, log_betas_ref, log_gammas_ref, log_etas_ref = \
                model_np.forward_backward(seq)

            log_alphas, log_evidence, log_betas, log_gammas, log_etas = \
                sess.run([model.log_alphas, model.log_evidence, model.log_betas, model.log_gammas, model.log_etas],
                         feed_dict={model.seq: seq_batch})
            log_alphas, log_evidence, log_betas, log_gammas, log_etas = \
                log_alphas[0], log_evidence[0], log_betas[0], log_gammas[0], log_etas[0]

            np.testing.assert_almost_equal(log_alphas, log_alphas_ref, decimal=2)
            np.testing.assert_almost_equal(log_evidence, log_evidence_ref, decimal=2)
            np.testing.assert_almost_equal(log_betas, log_betas_ref, decimal=2)
            np.testing.assert_almost_equal(log_gammas, log_gammas_ref, decimal=2)
            np.testing.assert_almost_equal(log_etas, log_etas_ref, decimal=2)
