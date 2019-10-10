import numpy as np
import matplotlib.pyplot as plt
from ..simple_gaussian import SimpleGaussian
from ..hmm_gaussian import HMMGaussian


sg = SimpleGaussian()
hmm = HMMGaussian(sg.A, sg.INIT, sg.MU, sg.COV)

# generate sequence
seq_length = 300

xs, zs = sg.generate_sequence(seq_length)

plt.title("observations")
plt.scatter(xs[:, 0][zs == 0], xs[:, 1][zs == 0], label="z=0")
plt.scatter(xs[:, 0][zs == 1], xs[:, 1][zs == 1], label="z=1")
plt.legend()
plt.show()

# calculate probabilities
log_alphas, log_evidence, log_betas, log_gammas, log_etas = hmm.forward_backward(xs)
alphas, gammas = np.exp(log_alphas), np.exp(log_gammas)

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
