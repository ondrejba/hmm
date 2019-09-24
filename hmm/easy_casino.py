import numpy as np


class Casino:
    """
    Occasionally dishonest casino from Machine Learning: A Probabilistic Perspective, Chapter 17.
    """

    Z_HONEST = 0
    Z_DISHONEST = 1

    A = np.array([
        [0.95, 0.05],
        [0.1, 0.9]
    ])

    PX = np.array([
        [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        [1 / 50, 1 / 50, 1 / 50, 1 / 50, 1 / 50, 45 / 50]
    ])

    INIT = np.array([0.95, 0.05])

    def __init__(self):

        self.z = int(np.random.choice([self.Z_HONEST, self.Z_DISHONEST], 1, p=self.INIT)[0])

    def transition(self):

        probs = self.A[self.z]
        self.z = int(np.random.choice([self.Z_HONEST, self.Z_DISHONEST], 1, p=probs)[0])

    def observe(self):

        if self.z == self.Z_HONEST:
            x = np.random.choice([0, 1, 2, 3, 4, 5], 1, p=self.PX[0])
        else:
            x = np.random.choice([0, 1, 2, 3, 4, 5], 1, p=self.PX[1])

        return int(x[0])
