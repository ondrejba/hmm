import numpy as np
import matplotlib.pyplot as plt
from ..casino import Casino
from ..hmm import HMM


casino = Casino()
hmm = HMM(Casino.A, Casino.PX, Casino.INIT)

# generate sequence
seq_length = 300

xs = [casino.observe()]
zs = [casino.z]

for i in range(seq_length - 1):
    casino.transition()
    xs.append(casino.observe())
    zs.append(casino.z)

# calculate probabilities
alphas, log_evidence = hmm.forward(xs)
betas = hmm.backward(xs)
gammas = alphas * betas
gammas = gammas / np.sum(gammas, axis=1)[:, np.newaxis]

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
