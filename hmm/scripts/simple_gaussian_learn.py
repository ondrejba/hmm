import numpy as np
import matplotlib.pyplot as plt
from ..simple_gaussian import SimpleGaussian
from ..hmm_gaussian import HMMGaussian


hmm = HMMGaussian()

# generate sequence
train_seq_length = 100
test_seq_length = 300
batch_size = 100

xs_batch = []
zs_batch = []

for j in range(batch_size):
    sg = SimpleGaussian()

    xs, zs = sg.generate_sequence(train_seq_length)

    xs_batch.append(xs)
    zs_batch.append(zs)

xs_batch = np.array(xs_batch)
zs_batch = np.array(zs_batch)

sg = SimpleGaussian()
test_xs, test_zs = sg.generate_sequence(test_seq_length)

num_hidden_states = len(np.unique(zs_batch))

# learn
hmm.initialize_em(2, 2)

for i in range(50):

    # learn
    print("step", i)
    print(hmm.A)
    print(hmm.init)
    print(hmm.mu)
    print(hmm.cov)
    print()

    ll = hmm.learn_em(xs_batch)
    print("log likelihood:", ll)
    print()

# calculate probabilities
log_alphas, log_evidence, log_betas, log_gammas, log_etas = hmm.forward_backward(test_xs)
alphas, gammas = np.exp(log_alphas), np.exp(log_gammas)

# plot alphas and gammas
plot_zs = np.array(test_zs)
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
