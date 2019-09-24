import numpy as np


class HMMMultinoulli:

    def __init__(self, A=None, PX=None, init=None):

        self.A = A
        self.PX = PX
        self.init = init

    def learn_fully_obs(self, xs, zs):

        _, N1 = np.unique(zs[:, 0], return_counts=True)
        num_hidden_states = len(np.unique(zs))
        num_observations = len(np.unique(xs))

        N = np.zeros((num_hidden_states, num_hidden_states), dtype=np.float32)

        for i in range(num_hidden_states):
            for j in range(num_hidden_states):
                for t in range(zs.shape[1] - 1):
                    N[i, j] += np.sum(np.bitwise_and(zs[:, t] == i, zs[:, t+1] == j))

        self.init = N1 / np.sum(N1)
        self.A = N / np.sum(N, axis=1)[:, np.newaxis]

        Nx = np.zeros((num_hidden_states, num_observations), dtype=np.float32)

        for i in range(num_hidden_states):
            for j in range(num_observations):
                Nx[i, j] = np.sum(np.bitwise_and(zs == i, xs == j))

        self.PX = Nx / np.sum(Nx, axis=1)[:, np.newaxis]

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
