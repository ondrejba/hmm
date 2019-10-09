from abc import ABC, abstractmethod
import numpy as np


class Env(ABC):

    def __init__(self):
        self.z = NotImplemented

    def generate_sequence(self, seq_length):

        xs = [self.observe()]
        zs = [self.z]

        for i in range(seq_length - 1):
            self.transition()
            xs.append(self.observe())
            zs.append(self.z)

        xs = np.array(xs)
        zs = np.array(zs)

        return xs, zs

    @abstractmethod
    def observe(self):
        pass

    @abstractmethod
    def transition(self):
        pass
