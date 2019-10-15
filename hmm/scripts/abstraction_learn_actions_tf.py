import os
import logging
import pickle
import click
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from ..hmm_gaussian_cat_actions_tf import HMMGaussianCatActionsTF
from .. import seq_utils

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel(logging.ERROR)


@click.command()
@click.option("--dimensionality", default=2, type=click.INT)
@click.option("--num-hidden-states", default=10, type=click.INT)
@click.option("--learning-rate", default=0.01, type=click.FLOAT)
@click.option("--num-steps", default=10000, type=click.INT)
@click.option("--validation-freq", default=200, type=click.INT)
@click.option("--minibatches", default=False, is_flag=True)
@click.option("--batch-size", default=100, type=click.INT)
@click.option("--gpu", default=None, type=click.STRING)
def main(dimensionality, num_hidden_states, learning_rate, num_steps, validation_freq, minibatches, batch_size, gpu):

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
        use_mask=True
    )

    hmm.setup()

    nn = KNeighborsClassifier(n_neighbors=1)
    nn.fit(seq_utils.flatten(seq), seq_utils.flatten(labels))

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

            _, log_likelihood, log_gammas = session.run(
                [hmm.opt_step, hmm.log_likelihood, hmm.log_gammas], feed_dict=feed_dict
            )
            print("step {:d}: {:.0f} ll".format(i, log_likelihood))

            if i % validation_freq == 0:

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

                print("accuracy: {:.2f}%".format(accuracy * 100))

                plt.subplot(2, 1, 1)
                plt.scatter(masked_flat_seq[:, 0], masked_flat_seq[:,  1], c=masked_flat_labels)

                plt.subplot(2, 1, 2)
                plt.scatter(masked_flat_seq[:, 0], masked_flat_seq[:, 1], c=predicted_labels_flat[seq_utils.flatten(masks)])

                plt.show()


if __name__ == "__main__":
    main()
