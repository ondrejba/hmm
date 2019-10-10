import numpy as np
import matplotlib.pyplot as plt
from ..hmm_gaussian import HMMGaussian


# load data
embeddings = np.load("./data/valid_embeddings.npy")
next_embeddings = np.load("./data/valid_next_embeddings.npy")
labels = np.load("./data/valid_labels.npy")

xs = np.stack([embeddings, next_embeddings], axis=1)
ys = labels

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
xs = np.reshape(xs, (-1, 20))
xs = pca.fit_transform(xs)
xs = np.reshape(xs, (-1, 2, 2))

print(xs[0])

# learn
hmm = HMMGaussian()
hmm.initialize_em(10, 2)

for i in range(50):

    # learn
    print("step", i)
    #print(hmm.A)
    #print(hmm.init)
    #print(hmm.mu)
    #print(hmm.cov)
    #print()

    ll = hmm.learn_em(xs[:100])
    print("log likelihood:", ll)
    print()

# calculate probabilities
log_alphas, log_evidence, log_betas, log_gammas, log_etas = hmm.forward_backward(xs[0])
alphas, gammas = np.exp(log_alphas), np.exp(log_gammas)

# plot alphas and gammas
plot_zs = np.array(ys)
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
