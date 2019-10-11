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

# learn
hmm = HMMGaussian()
hmm.initialize_em(10, 2)

for i in range(50):

    # learn
    print("step", i)
    #print(hmm.A)
    print(hmm.enabled)
    #print(hmm.init)
    #print(hmm.mu)
    #print(hmm.cov)
    #print()


    ll = hmm.learn_em(xs)
    print("log likelihood:", ll)
    print()

# calculate probabilities
plot_xs = xs.reshape((-1, 2))
plt.scatter(plot_xs[:, 0], plot_xs[:, 1])
plt.scatter(hmm.mu[:, 0], hmm.mu[:, 1])
plt.show()
