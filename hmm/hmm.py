import numpy as np


class HMMMultinoulli:

    def __init__(self, A=None, PX=None, init=None):

        self.A = A
        self.PX = PX
        self.init = init

        if init is not None and PX is not None:
            self.num_hidden_states = len(self.init)
            self.num_observations = self.PX.shape[1]

    def learn_fully_obs(self, xs, zs):

        _, N1 = np.unique(zs[:, 0], return_counts=True)
        self.num_hidden_states = len(np.unique(zs))
        self.num_observations = len(np.unique(xs))

        N = np.zeros((self.num_hidden_states, self.num_hidden_states), dtype=np.float32)

        for i in range(self.num_hidden_states):
            for j in range(self.num_hidden_states):
                for t in range(zs.shape[1] - 1):
                    N[i, j] += np.sum(np.bitwise_and(zs[:, t] == i, zs[:, t+1] == j))

        self.init = N1 / np.sum(N1)
        self.A = N / np.sum(N, axis=1)[:, np.newaxis]

        Nx = np.zeros((self.num_hidden_states, self.num_observations), dtype=np.float32)

        for i in range(self.num_hidden_states):
            for j in range(self.num_observations):
                Nx[i, j] = np.sum(np.bitwise_and(zs == i, xs == j))

        self.PX = Nx / np.sum(Nx, axis=1)[:, np.newaxis]

    def learn_em(self, xs, num_hidden_states, num_steps=10):

        self.num_hidden_states = num_hidden_states
        self.num_observations = len(np.unique(xs))

        batch_size = len(xs)

        # init
        self.init = np.random.dirichlet([1.0] * self.num_hidden_states)
        self.A = np.random.dirichlet([1.0] * self.num_hidden_states * self.num_hidden_states)\
            .reshape((self.num_hidden_states, self.num_hidden_states))
        self.PX = np.random.dirichlet([1.0] * self.num_observations, size=self.num_hidden_states)

        for step_idx in range(num_steps):

            # E step
            gammas_batch = []
            etas_batch = []

            for batch_idx in range(batch_size):
                _, _, _, gammas, etas = self.forward_backward(xs[batch_idx])

                gammas_batch.append(gammas)
                etas_batch.append(etas)

            gammas_batch = np.array(gammas_batch)
            etas_batch = np.array(etas_batch)

            N1 = np.zeros(self.num_hidden_states, dtype=np.float64)

            for i in range(self.num_hidden_states):
                N1[i] = np.sum(gammas_batch[:, 0, i])

            Nj = np.zeros(self.num_hidden_states, dtype=np.float64)

            for i in range(self.num_hidden_states):
                Nj[i] = np.sum(gammas_batch[:, :, i])

            Njk = np.sum(etas_batch, axis=(0, 1))

            # M step
            self.A = Njk / np.sum(Njk)
            self.init = N1 / np.sum(N1)

            Mjl = np.zeros((self.num_hidden_states, self.num_observations), dtype=np.float64)

            for i in range(self.num_observations):
                mask = (xs == i).astype(np.float64)
                tmp = np.sum(gammas_batch * mask[:, :, np.newaxis], axis=(0, 1))
                Mjl[:, i] = tmp

            self.PX = Mjl / Nj[:, np.newaxis]

    def forward_backward(self, seq):

        alphas, log_evidence = self.forward(seq)
        betas = self.backward(seq)

        px = self.condition(seq)

        gammas = alphas * betas
        gammas = gammas / np.sum(gammas, axis=1)[:, np.newaxis]

        etas = []

        for t in range(1, len(seq)):

            a1 = alphas[t - 1]
            a2 = px[:, t]
            a3 = betas[t]

            eta = self.A * a1[:, np.newaxis] * a2[np.newaxis, :] * a3[np.newaxis, :]
            etas.append(eta)

        etas = np.array(etas)
        etas = etas / np.sum(etas, axis=(1, 2))[:, np.newaxis, np.newaxis]

        return alphas, log_evidence, betas, gammas, etas

    def forward(self, seq):

        px = self.condition(seq)

        alpha_1 = px[:, 0] * self.init
        alpha_1, Z_1 = self.normalize(alpha_1)

        alphas = [alpha_1]
        Zs = [Z_1]

        for t in range(1, len(seq)):
            alpha_t = px[:, t] * np.matmul(np.transpose(self.A, axes=[1, 0]), alphas[-1])
            alpha_t, Zt = self.normalize(alpha_t)

            alphas.append(alpha_t)
            Zs.append(Zt)

        alphas = np.array(alphas)
        Zs = np.array(Zs)

        log_evidence = np.sum(np.log(Zs))

        return alphas, log_evidence

    def backward(self, seq):

        px = self.condition(seq)

        betas = [[1.0] * px.shape[0]]

        for t in reversed(range(0, len(seq) - 1)):
            beta_t = np.matmul(self.A, px[:, t] * betas[0])
            betas.insert(0, beta_t)

        betas = np.array(betas)

        return betas

    def condition(self, seq):
        return self.PX[:, seq]

    def normalize(self, alpha):
        s = np.sum(alpha)
        return alpha / s, s
