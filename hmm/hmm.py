import numpy as np


class HMM:

    def __init__(self, A, PX, init):

        self.A = A
        self.PX = PX
        self.init = init

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
