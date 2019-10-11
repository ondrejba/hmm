import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from ..simple_gaussian import SimpleGaussian
from ..hmm_gaussian_tf import HMMGaussianTF

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel(logging.ERROR)

seq_length = 300
batch_size = 100
learning_rate = 0.01
num_steps = 500

hmm = HMMGaussianTF(2, 2, seq_length=seq_length, learning_rate=learning_rate)

# generate sequence
xs_batch = []
zs_batch = []

for j in range(batch_size):
    sg = SimpleGaussian()

    xs = [sg.observe()]
    zs = [sg.z]

    for i in range(seq_length - 1):
        sg.transition()
        xs.append(sg.observe())
        zs.append(sg.z)

    xs_batch.append(xs)
    zs_batch.append(zs)

xs_batch = np.array(xs_batch)
zs_batch = np.array(zs_batch)

num_hidden_states = len(np.unique(zs_batch))

# learn
hmm.setup()

with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    for i in range(num_steps):
        _, log_likelihood = session.run([hmm.opt_step, hmm.log_likelihood], feed_dict={hmm.seq: xs_batch})
        print("step {:d}: {:.0f} ll".format(i, log_likelihood))

    # calculate probabilities
    log_alphas, log_gammas = session.run([hmm.log_alphas, hmm.log_gammas], feed_dict={hmm.seq: xs_batch})

alphas = np.exp(log_alphas[0])
gammas = np.exp(log_gammas[0])

# plot alphas and gammas
plot_zs = np.array(zs_batch[0])
plot_alphas = alphas[:, 1]
plot_gammas = gammas[:, 1]
plot_xs = np.linspace(1, len(plot_zs), num=len(plot_zs))

plt.figure(figsize=(12, 9))

plt.subplot(2, 1, 1)
plt.title("filtering")
plt.plot(plot_xs, 1 - plot_zs, label="z")
plt.plot(plot_xs, plot_alphas, label="P(z) = 1")
plt.legend()

plt.subplot(2, 1, 2)
plt.title("smoothing")
plt.plot(plot_xs, 1 - plot_zs, label="z")
plt.plot(plot_xs, plot_gammas, label="P(z) = 1")
plt.legend()
plt.show()
