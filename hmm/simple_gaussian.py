import numpy as np


class SimpleGaussian:

    A = np.array([
        [0.85, 0.15],
        [0.2, 0.8]
    ])
    INIT = np.array([0.7, 0.3])
    ZS = [0, 1]

    MU = np.array([[1.0, 1.0], [-1.0, -1.0]])
    COV = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])

    def __init__(self):

        self.z = int(np.random.choice(self.ZS, 1, p=self.INIT)[0])

    def transition(self):

        probs = self.A[self.z]
        self.z = int(np.random.choice(self.ZS, 1, p=probs)[0])

    def observe(self):

        if self.z == self.ZS[0]:
            x = np.random.multivariate_normal(self.MU[0], self.COV[0])
        else:
            x = np.random.multivariate_normal(self.MU[1], self.COV[1])

        return x
