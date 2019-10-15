import numpy as np
import tensorflow as tf


class HMMGaussianCatActionsTF:

    def __init__(self, num_hidden_states, dimensionality, num_actions, seq_length, learning_rate=0.01, full_cov=False,
                 use_mask=False):

        self.num_hidden_states = num_hidden_states
        self.dimensionality = dimensionality
        self.num_actions = num_actions
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.full_cov = full_cov
        self.use_mask = use_mask

    def setup_variables(self):

        self.init_v = tf.Variable(np.random.normal(0, 1, size=self.num_hidden_states), trainable=True, dtype=tf.float32)
        self.init = tf.nn.softmax(self.init_v, axis=0)
        self.A_v = tf.Variable(np.random.normal(
            0, 1, size=(self.num_actions, self.num_hidden_states, self.num_hidden_states)
        ), trainable=True, dtype=tf.float32)
        self.A = tf.nn.softmax(self.A_v, axis=2)
        self.mu = tf.Variable(
            np.random.normal(0, 1, size=(self.num_hidden_states, self.dimensionality)), trainable=True, dtype=tf.float32
        )

        if self.full_cov:
            self.cov_v = tf.Variable(
                np.tile(np.diag([1.0] * self.dimensionality)[np.newaxis, :, :], reps=(self.num_hidden_states, 1, 1)),
                trainable=True, dtype=tf.float32
            )
            self.cov = tf.nn.softplus(self.cov_v)
            self.dists = tf.contrib.distributions.MultivariateNormalFullCovariance(
                loc=self.mu, covariance_matrix=self.cov
            )
        else:
            self.cov_v = tf.Variable(
                np.random.normal(0, 1, size=(self.num_hidden_states, self.dimensionality)), trainable=True,
                dtype=tf.float32
            )
            self.cov = tf.nn.softplus(self.cov_v)
            self.dists = tf.contrib.distributions.MultivariateNormalDiag(
                loc=self.mu, scale_diag=tf.sqrt(self.cov)
            )

        self.log_init = tf.log(self.init)
        self.log_A = tf.log(self.A)

    def setup_placeholders(self):

        self.seq = tf.placeholder(tf.float32, shape=(None, self.seq_length, self.dimensionality), name="seq_pl")
        self.actions = tf.placeholder(tf.int32, shape=(None, self.seq_length - 1), name="actions_pl")
        self.batch_size = tf.shape(self.seq)[0]
        self.log_condition_seq = self.condition(self.seq)
        self.gather_log_A = tf.gather(self.log_A, self.actions)

        if self.use_mask:
            self.mask = tf.placeholder(tf.bool, shape=(None, self.seq_length), name="mask_pl")
            self.float_mask = tf.cast(self.mask, tf.float32)
        else:
            self.float_mask = tf.ones((self.batch_size, self.seq_length), dtype=tf.float32, name="fixed_mask")

    def setup(self):

        self.setup_variables()
        self.setup_placeholders()
        self.log_likelihood = self.likelihood()

        self.opt_step = tf.train.AdamOptimizer(self.learning_rate).minimize(- self.log_likelihood)

    def likelihood(self):

        # forward backward
        self.log_alphas, self.log_evidence, self.log_betas, self.log_gammas, self.log_etas = \
            self.forward_backward()

        # counts
        log_N1 = tf.reduce_logsumexp(self.log_gammas[:, 0, :], axis=0)

        # log likelihood
        log_likelihood = \
            tf.reduce_sum(tf.exp(log_N1) * tf.log(self.init)) + \
            tf.reduce_sum(
                (tf.exp(self.log_etas) * self.gather_log_A) * self.float_mask[:, 1:, tf.newaxis, tf.newaxis]
            ) + \
            tf.reduce_sum(
                (tf.exp(self.log_gammas) * self.log_condition_seq) * self.float_mask[:, :, tf.newaxis]
            )

        return log_likelihood

    def forward_backward(self):

        log_alphas, log_evidence = self.forward()
        log_betas = self.backward()

        log_gammas = log_alphas + log_betas
        log_gammas = log_gammas - tf.reduce_logsumexp(log_gammas, axis=2)[:, :, tf.newaxis]

        log_etas = self.gather_log_A + (self.log_condition_seq[:, 1:, :] +
            log_betas[:, 1:, :])[:, :, tf.newaxis, :] + log_alphas[:, :-1, :][:, :, :, tf.newaxis]
        log_etas = log_etas - tf.reduce_logsumexp(log_etas, axis=(2, 3))[:, :, tf.newaxis, tf.newaxis]

        return log_alphas, log_evidence, log_betas, log_gammas, log_etas

    def forward(self):

        init_log_alpha = self.log_condition_seq[:, 0, :] + self.log_init[tf.newaxis, :]
        init_log_alpha, init_log_Zs = self.log_normalize(init_log_alpha)

        def fc(last, t):

            log_alphas_t_minus_one, log_Zs_t_minus_one = last

            log_alpha_t = self.log_condition_seq[:, t, :] + tf.reduce_logsumexp(
                self.gather_log_A[:, t-1, :, :] + log_alphas_t_minus_one[:, :, tf.newaxis], axis=1
            )

            log_alpha_t, log_Zt = self.log_normalize(log_alpha_t)

            return log_alpha_t, log_Zt

        log_alphas, log_Zs = tf.scan(elems=tf.constant(list(range(1, self.seq_length))), fn=fc,
                                     initializer=(init_log_alpha, init_log_Zs))
        log_alphas = tf.transpose(log_alphas, perm=[1, 0, 2])
        log_Zs = tf.transpose(log_Zs, perm=[1, 0])

        log_alphas = tf.concat([init_log_alpha[:, tf.newaxis, :], log_alphas], axis=1)
        log_Zs = tf.concat([init_log_Zs[:, tf.newaxis], log_Zs], axis=1)

        log_evidence = tf.reduce_sum(log_Zs, axis=1)

        return log_alphas, log_evidence

    def backward(self):

        init_betas = tf.zeros((self.batch_size, self.num_hidden_states), dtype=tf.float32)

        def fc(log_beta_t_plus_one, t):

            log_beta_t = tf.reduce_logsumexp(
                self.gather_log_A[:, t, :, :] +
                (self.log_condition_seq[:, t, :] + log_beta_t_plus_one)[:, tf.newaxis, :], axis=2
            )

            return log_beta_t

        log_betas = tf.scan(elems=tf.constant(list(range(0, self.seq_length - 1))),
                            fn=fc, initializer=init_betas, reverse=True)
        log_betas = tf.transpose(log_betas, perm=[1, 0, 2])
        log_betas = tf.concat([log_betas, init_betas[:, tf.newaxis, :]], axis=1)

        return log_betas

    def condition(self, seq):

        seq = tf.reshape(seq, shape=(-1, self.dimensionality))

        log_px = self.dists.log_prob(seq[:, tf.newaxis, :])
        log_px = tf.reshape(log_px, shape=(-1, self.seq_length, self.num_hidden_states))

        return log_px

    def log_normalize(self, alpha):

        s = tf.reduce_logsumexp(alpha, axis=-1)
        return alpha - s[..., tf.newaxis], s
