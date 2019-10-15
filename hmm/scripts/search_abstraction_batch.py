import os
import pickle
import numpy as np
from ..runners import abstraction_learn_actions_tf

lrs = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
dims = [2, 3, 4, 8, 16, 32]
mu_sds = [0.1, 1.0, 5.0]
cov_sds = [0.1, 1.0, 5.0]
runs = 10

results = dict()

if not os.path.isdir("./results"):
    os.makedirs("./results")

for i, mu_sd in enumerate(mu_sds):
    for j, cov_sd in enumerate(cov_sds):
        for k, lr in enumerate(lrs):
            for l, dim in enumerate(dims):
                for run_idx in range(10):

                    if (mu_sd, cov_sd, lr, dim, run_idx) in results:
                        continue

                    best_accuracy, _ = abstraction_learn_actions_tf.main(
                        dim, 10, lr, 500, 100, False, 100, mu_sd, cov_sd, False, "1"
                    )

                    if np.any(np.isnan(best_accuracy)):
                        best_accuracy = 0.0

                    results[mu_sd, cov_sd, lr, dim, run_idx] = best_accuracy

                    with open("./results/search_abstraction_batch.pickle", "wb") as file:
                        pickle.dump(results, file)
