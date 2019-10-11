import numpy as np
import tensorflow as tf


class HMMGaussianTF:

    def __init__(self, num_hidden_states, dimensionality, seq_length, learning_rate=0.01):

        self.num_hidden_states = num_hidden_states
        self.dimensionality = dimensionality
        self.seq_length = seq_length
        self.learning_rate = learning_rate

    def setup_variables(self):

        self.init = tf.Variable(np.random.dirichlet([1.0] * self.num_hidden_states), trainable=True, dtype=tf.float32)
        self.A = tf.Variable(np.random.dirichlet([1.0] * self.num_hidden_states * self.num_hidden_states) \
            .reshape((self.num_hidden_states, self.num_hidden_states)), trainable=True, dtype=tf.float32)
        self.mu = tf.Variable(
            np.random.normal(0, 1, size=(self.num_hidden_states, self.dimensionality)), trainable=True, dtype=tf.float32
        )
        self.cov = tf.Variable(
            np.tile(np.diag([1.0] * self.dimensionality)[np.newaxis, :, :], reps=(self.num_hidden_states, 1, 1)),
            trainable=True, dtype=tf.float32
        )

        self.dists = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=self.mu, covariance_matrix=self.cov)

    def setup_placeholders(self):

        self.seq = tf.placeholder(tf.float32, shape=(None, self.seq_length, self.dimensionality), name="seq_pl")
        self.batch_size = tf.shape(self.seq)[0]
        self.log_condition_seq = self.condition(self.seq)

    def setup(self):

        self.setup_variables()
        self.setup_placeholders()
        self.log_likelihood = self.likelihood()

        self.opt_step = tf.train.AdamOptimizer(self.learning_rate).minimize(- self.log_likelihood)

    def likelihood(self):

        # forward backward
        _, _, _, gammas_batch, etas_batch = self.forward_backward()

        # counts
        N1 = tf.reduce_sum(gammas_batch[:, 0, :], axis=0)
        Njk = tf.reduce_sum(etas_batch, axis=(0, 1))

        # log likelihood
        log_likelihood = tf.reduce_sum(N1 * tf.log(self.init)) + \
            tf.reduce_sum(Njk * tf.log(self.A)) + tf.reduce_sum(gammas_batch * self.log_condition_seq)

        return log_likelihood

    def forward_backward(self):

        log_alphas, log_evidence = self.forward()
        log_betas = self.backward()

        log_gammas = log_alphas + log_betas
        log_gammas = log_gammas - tf.reduce_logsumexp(log_gammas, axis=1)[:, tf.newaxis]
        gammas = tf.exp(log_gammas)

        log_A = tf.log(self.A)

        log_etas = log_A[tf.newaxis, tf.newaxis, :, :] + (self.log_condition_seq[:, 1:, :] +
            log_betas[:, 1:, :])[:, :, tf.newaxis, :] + log_alphas[:, :-1, :][:, :, :, tf.newaxis]
        log_etas = log_etas - tf.reduce_logsumexp(log_etas, axis=(1, 2))[:, tf.newaxis, tf.newaxis]
        etas = tf.exp(log_etas)

        return log_alphas, log_evidence, log_betas, gammas, etas

    def forward(self):

        log_init = tf.log(self.init)
        log_A = tf.log(self.A)

        log_alpha_1 = self.log_condition_seq[:, 0, :] + log_init[tf.newaxis, :]
        log_alpha_1, log_Z_1 = self.log_normalize(log_alpha_1)

        log_alphas = [log_alpha_1]
        log_Zs = [log_Z_1]

        for t in range(1, self.seq_length):

            log_alpha_t = self.log_condition_seq[:, t, :] + tf.reduce_logsumexp(
                log_A[tf.newaxis, :, :] + log_alphas[-1][:, :, tf.newaxis], axis=1
            )

            log_alpha_t, log_Zt = self.log_normalize(log_alpha_t)

            log_alphas.append(log_alpha_t)
            log_Zs.append(log_Zt)

        log_alphas = tf.stack(log_alphas, axis=1)
        log_Zs = tf.stack(log_Zs, axis=1)

        log_evidence = tf.reduce_sum(log_Zs, axis=1)

        return log_alphas, log_evidence

    def backward(self):

        log_betas = [tf.zeros((self.batch_size, self.num_hidden_states), dtype=tf.float32)]
        log_A = tf.log(self.A)
        log_px = self.log_condition_seq

        for t in reversed(range(0, self.seq_length - 1)):

            log_beta_t = tf.reduce_logsumexp(
                log_A[tf.newaxis, :, :] + (log_px[:, t, :] + log_betas[0])[:, tf.newaxis, :], axis=2
            )
            log_betas.insert(0, log_beta_t)

        log_betas = tf.stack(log_betas, axis=1)

        return log_betas

    def condition(self, seq):

        seq = tf.reshape(seq, shape=(-1, self.dimensionality))

        log_px = self.dists.log_prob(seq[:, tf.newaxis, :])
        log_px = tf.reshape(log_px, shape=(-1, self.seq_length, self.num_hidden_states))

        return log_px

    def log_normalize(self, alpha):

        s = tf.reduce_logsumexp(alpha, axis=-1)
        return alpha - s[..., tf.newaxis], s
