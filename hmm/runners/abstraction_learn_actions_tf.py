import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from ..hmm_gaussian_cat_actions_tf import HMMGaussianCatActionsTF
from .. import seq_utils


def main(dimensionality, num_hidden_states, learning_rate, num_steps, validation_freq, minibatches, batch_size,
         mu_init_sd, cov_init_sd, show_graphs, gpu):

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    dataset_path = "data/new_train_dataset.pickle"
    with open(dataset_path, "rb") as file:
        dataset = pickle.load(file)

    seq = dataset["embeddings"]
    labels = dataset["state_labels"]
    actions = dataset["actions"][:, :-1]
    masks = dataset["masks"]

    seq_length = 20
    num_actions = 4

    if dimensionality < seq.shape[2]:
        seq = seq_utils.project_embeddings(seq, masks, dimensionality)

    masked_flat_seq = seq_utils.mask_embeddings(seq, masks)
    masked_flat_labels = seq_utils.mask_embeddings(labels, masks)

    hmm = HMMGaussianCatActionsTF(
        num_hidden_states, dimensionality, num_actions, seq_length=seq_length, learning_rate=learning_rate,
        use_mask=True, mu_init_sd=mu_init_sd, cov_init_sd=cov_init_sd
    )

    hmm.setup()

    nn = KNeighborsClassifier(n_neighbors=1)
    nn.fit(seq_utils.flatten(seq), seq_utils.flatten(labels))

    accuracies = []

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        for i in range(num_steps):

            if minibatches:
                epoch_size = seq.shape[0] // batch_size
                epoch_step = i % epoch_size
                b_idx = np.index_exp[epoch_step * batch_size: (epoch_step + 1) * batch_size]

                feed_dict = {
                    hmm.seq: seq[b_idx],
                    hmm.actions: actions[b_idx],
                    hmm.mask: masks[b_idx]
                }
            else:
                feed_dict = {
                    hmm.seq: seq,
                    hmm.actions: actions,
                    hmm.mask: masks
                }

            _, log_likelihood = session.run(
                [hmm.opt_step, hmm.log_likelihood], feed_dict=feed_dict
            )

            if np.isnan(log_likelihood):
                return np.nan

            print("step {:d}: {:.0f} ll".format(i, log_likelihood))

            if i % validation_freq == 0 or i == num_steps - 1:

                log_gammas = session.run(hmm.log_gammas, feed_dict={
                    hmm.seq: seq,
                    hmm.actions: actions,
                    hmm.mask: masks
                })

                if np.any(np.isnan(log_gammas)):
                    return np.nan

                gammas = np.exp(log_gammas)
                assignment = np.argmax(gammas, axis=2)

                observations = session.run(hmm.dists.sample(assignment.shape))

                assignment_flat = seq_utils.flatten(assignment)
                observations_flat = seq_utils.flatten(observations)

                select_observations_flat = observations_flat[range(len(assignment_flat)), assignment_flat]
                predicted_labels_flat = nn.predict(select_observations_flat)

                masks_flat = seq_utils.flatten(masks)
                labels_flat = seq_utils.flatten(labels)

                accuracy = np.mean((predicted_labels_flat == labels_flat)[masks_flat])
                accuracies.append(accuracy)

                print("accuracy: {:.2f}%".format(accuracy * 100))

                if show_graphs:
                    plt.subplot(2, 1, 1)
                    plt.title("coarsest abstraction")
                    plt.scatter(masked_flat_seq[:, 0], masked_flat_seq[:,  1], c=masked_flat_labels)
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlabel("PCA1")
                    plt.ylabel("PCA2")

                    plt.subplot(2, 1, 2)
                    plt.title("predicted abstraction")
                    plt.scatter(masked_flat_seq[:, 0], masked_flat_seq[:, 1], c=predicted_labels_flat[seq_utils.flatten(masks)])
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlabel("PCA1")
                    plt.ylabel("PCA2")

                    plt.show()

    return np.max(accuracies), np.argmax(accuracies) * validation_freq
