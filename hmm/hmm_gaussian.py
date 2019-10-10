import numpy as np
from scipy import special
from scipy.stats import multivariate_normal


class HMMGaussian:

    def __init__(self, A=None, init=None, mu=None, cov=None):

        self.A = A
        self.init = init
        self.mu = mu
        self.cov = cov

        if init is not None and self.mu is not None:
            self.num_hidden_states = len(self.init)
            self.dimensionality = self.mu.shape[1]

    def learn_fully_obs(self, xs, zs):

        _, N1 = np.unique(zs[:, 0], return_counts=True)
        _, Nk = np.unique(zs, return_counts=True)
        self.num_hidden_states = len(np.unique(zs))
        self.dimensionality = xs.shape[2]

        Njl = np.zeros((self.num_hidden_states, self.num_hidden_states), dtype=np.float32)

        for i in range(self.num_hidden_states):
            for j in range(self.num_hidden_states):
                for t in range(zs.shape[1] - 1):
                    Njl[i, j] += np.sum(np.bitwise_and(zs[:, t] == i, zs[:, t+1] == j))

        self.init = N1 / np.sum(N1)
        self.A = Njl / np.sum(Njl, axis=1)[:, np.newaxis]

        self.mu = np.zeros((self.num_hidden_states, self.dimensionality), dtype=np.float64)
        mean_xx_bar = np.zeros((self.num_hidden_states, self.dimensionality, self.dimensionality), dtype=np.float64)

        for i in range(self.num_hidden_states):
            mask = zs == i
            self.mu[i, :] = np.mean(xs[mask], axis=0)
            mean_xx_bar[i, :, :] = np.mean(xs[mask][:, :, np.newaxis] * xs[mask][:, np.newaxis, :], axis=0)

        self.cov = mean_xx_bar - self.mu[:, :, np.newaxis] * self.mu[:, np.newaxis, :]

    def initialize_em(self, num_hidden_states, dimensionality):

        self.num_hidden_states = num_hidden_states
        self.dimensionality = dimensionality

        self.init = np.random.dirichlet([1.0] * self.num_hidden_states)
        self.A = np.random.dirichlet([1.0] * self.num_hidden_states * self.num_hidden_states) \
            .reshape((self.num_hidden_states, self.num_hidden_states))
        self.mu = np.random.normal(0, 1, size=(self.num_hidden_states, self.dimensionality))
        self.cov = np.tile(np.diag([0.1] * self.dimensionality)[np.newaxis, :, :],
                           reps=(self.num_hidden_states, 1, 1))
        self.enabled = np.array([True] * self.num_hidden_states)

    def learn_em(self, xs):

        batch_size = len(xs)

        # E step
        log_gammas_batch = []
        log_etas_batch = []

        for batch_idx in range(batch_size):
            _, _, _, log_gammas, log_etas = self.forward_backward(xs[batch_idx])

            log_gammas_batch.append(log_gammas)
            log_etas_batch.append(log_etas)

        log_gammas_batch = np.array(log_gammas_batch)
        log_etas_batch = np.array(log_etas_batch)

        log_N1 = np.zeros(self.num_hidden_states, dtype=np.float64)

        for i in range(self.num_hidden_states):
            log_N1[i] = special.logsumexp(log_gammas_batch[:, 0, i])

        log_Nj = np.zeros(self.num_hidden_states, dtype=np.float64)

        for i in range(self.num_hidden_states):
            log_Nj[i] = special.logsumexp(log_gammas_batch[:, :, i])

        log_Njk = special.logsumexp(log_etas_batch, axis=(0, 1))

        # M step
        log_A = log_Njk - special.logsumexp(log_Njk, axis=1)[:, np.newaxis]
        log_init = log_N1 - special.logsumexp(log_N1)

        self.A = np.exp(log_A)
        self.init = np.exp(log_init)

        mean_xx_bar = np.zeros((self.num_hidden_states, self.dimensionality, self.dimensionality), dtype=np.float64)

        for i in range(self.num_hidden_states):
            self.mu[i, :] = np.sum(xs * np.exp(log_gammas_batch[:, :, i:i+1] - log_Nj[i]), axis=(0, 1))
            mean_xx_bar[i, :, :] = np.sum(np.exp(log_gammas_batch[:, :, i][:, :, np.newaxis, np.newaxis] - log_Nj[i]) *
                                          xs[:, :, :, np.newaxis] * xs[:, :, np.newaxis, :], axis=(0, 1))

        self.cov = mean_xx_bar - self.mu[:, :, np.newaxis] * self.mu[:, np.newaxis, :]

        self.enabled[:] = True
        for i in range(self.num_hidden_states):
            if not self.is_valid(self.cov[i]):
                self.enabled[i] = False

        # log likelihood
        log_px = self.condition(xs)
        log_px = np.transpose(log_px, axes=[1, 2, 0])

        log_likelihood = np.sum(np.exp(log_N1) * np.log(self.init)) + \
                         np.sum(np.exp(log_Njk) * np.log(self.A)[np.newaxis, np.newaxis, :, :]) + \
                         np.sum(np.exp(log_gammas_batch) * log_px)

        return log_likelihood

    def forward_backward(self, seq):

        log_alphas, log_evidence = self.forward(seq)
        log_betas = self.backward(seq)

        log_px = self.condition(seq)

        log_gammas = log_alphas + log_betas
        log_gammas = log_gammas - special.logsumexp(log_gammas, axis=1)[:, np.newaxis]

        log_etas = []
        log_A = np.log(self.A)

        for t in range(1, len(seq)):
            log_eta = log_A + (log_px[:, t] + log_betas[t])[np.newaxis, :] + log_alphas[t - 1][:, np.newaxis]
            log_etas.append(log_eta)

        log_etas = np.array(log_etas)
        log_etas = log_etas - special.logsumexp(log_etas, axis=(1, 2))[:, np.newaxis, np.newaxis]

        return log_alphas, log_evidence, log_betas, log_gammas, log_etas

    def forward(self, seq):

        log_A = np.log(self.A)
        log_init = np.log(self.init)
        log_px = self.condition(seq)

        log_alpha_1 = log_px[:, 0] + log_init
        log_alpha_1, log_Z_1 = self.log_normalize(log_alpha_1)

        log_alphas = [log_alpha_1]
        log_Zs = [log_Z_1]

        for t in range(1, len(seq)):

            log_alpha_t = log_px[:, t] + special.logsumexp(log_A + log_alphas[-1][:, np.newaxis], axis=0)
            log_alpha_t, log_Zt = self.log_normalize(log_alpha_t)

            log_alphas.append(log_alpha_t)
            log_Zs.append(log_Zt)

        log_alphas = np.array(log_alphas)
        log_Zs = np.array(log_Zs)

        log_evidence = np.sum(log_Zs)

        return log_alphas, log_evidence

    def backward(self, seq):

        log_px = self.condition(seq)

        log_betas = [np.array([0.0] * log_px.shape[0])]
        log_A = np.log(self.A)

        for t in reversed(range(0, len(seq) - 1)):

            log_beta_t = special.logsumexp(
                log_A[:, :] + (log_px[:, t] + log_betas[0])[np.newaxis, :], axis=1
            )
            log_betas.insert(0, log_beta_t)

        log_betas = np.array(log_betas)

        return log_betas

    def condition(self, seq):

        log_px = []

        for i in range(self.num_hidden_states):

            if not self.enabled[i]:
                tmp_px = np.zeros(seq.shape[:-1], dtype=np.float32) - 10.0
            else:
                tmp_px = multivariate_normal.logpdf(seq, mean=self.mu[i], cov=self.cov[i])

            log_px.append(tmp_px)

        log_px = np.array(log_px)
        return log_px

    def normalize(self, alpha):
        s = np.sum(alpha)
        return alpha / s, s

    def log_normalize(self, alpha):
        s = special.logsumexp(alpha)
        return alpha - s, s

    def is_valid(self, x):
        return np.all(np.linalg.eigvals(x) >= 0) and np.linalg.matrix_rank(x) == x.shape[0]
