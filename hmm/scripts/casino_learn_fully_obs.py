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

# learn
hmm.learn_fully_obs(xs_batch, zs_batch)

# print results
for key1, value1 in zip(["init", "A", "PX"], [[Casino.INIT, hmm.init], [Casino.A, hmm.A], [Casino.PX, hmm.PX]]):
    for key2, value2 in zip(["real", "learned"], value1):
        print("{} {}:".format(key2, key1))
        print(value2)

    print("mean absolute error: {:.4f}".format(np.mean(np.abs(value1[0] - value1[1]))))
    print()

# try filtering and smoothing with the learned model
xs = xs_batch[0]
zs = zs_batch[0]

alphas, log_evidence, betas, gammas, etas = hmm.forward_backward(xs)

# plot alphas and gammas
plot_zs = np.array(zs)
plot_alphas = alphas[:, 1]
plot_gammas = gammas[:, 1]
plot_xs = np.linspace(1, len(plot_zs), num=len(plot_zs))

plt.figure(figsize=(12, 9))

plt.subplot(2, 1, 1)
plt.title("filtering")
plt.plot(plot_xs, plot_zs, label="z")
plt.plot(plot_xs, plot_alphas, label="P(z) = 1")
plt.legend()

plt.subplot(2, 1, 2)
plt.title("smoothing")
plt.plot(plot_xs, plot_zs, label="z")
plt.plot(plot_xs, plot_gammas, label="P(z) = 1")
plt.legend()
plt.show()
