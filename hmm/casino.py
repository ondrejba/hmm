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

    def __init__(self):

        self.z = self.Z_HONEST

    def transition(self):

        probs = self.A[self.z]
        self.z = int(np.random.choice([self.Z_HONEST, self.Z_DISHONEST], 1, p=probs)[0])

    def observe(self):

        if self.z == self.Z_HONEST:
            return np.random.choice([1, 2, 3, 4, 5, 6], 1, p=[1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
        else:
            return np.random.choice([1, 2, 3, 4, 5, 6], 1, p=[1/10, 1/10, 1/10, 1/10, 1/10, 5/10])
