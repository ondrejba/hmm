import numpy as np
import matplotlib.pyplot as plt
from ..casino import Casino
from ..hmm import HMMMultinoulli


hmm = HMMMultinoulli(Casino.A, Casino.PX, Casino.INIT)

# generate sequence
seq_length = 300
batch_size = 100

xs_batch = []
zs_batch = []

for j in range(batch_size):
    casino = Casino()

    xs = [casino.observe()]
    zs = [casino.z]

    for i in range(seq_length - 1):
        casino.transition()
        xs.append(casino.observe())
        zs.append(casino.z)

    xs_batch.append(xs)
    zs_batch.append(zs)

xs_batch = np.array(xs_batch)
zs_batch = np.array(zs_batch)

num_hidden_states = len(np.unique(zs_batch))

# learn
hmm.learn_em(xs_batch, num_hidden_states, num_steps=10)
