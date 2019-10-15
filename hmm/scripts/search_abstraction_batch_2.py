import os
import pickle
import numpy as np
from ..runners import abstraction_learn_actions_tf

lrs = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
dims = [2, 3, 4, 8, 16, 32]
num_hidden_states_list = [10, 20, 50, 100]
runs = 10

results = dict()
results_path = "./results/search_abstraction_batch_2.pickle"

if not os.path.isdir("./results"):
    os.makedirs("./results")

if os.path.isfile(results_path):
    results = pickle.load(results_path)

for i, num_hidden_states in enumerate(num_hidden_states_list):
    for j, lr in enumerate(lrs):
        for k, dim in enumerate(dims):
            for run_idx in range(10):

                if (num_hidden_states, lr, dim, run_idx) in results:
                    continue

                best_accuracy, _ = abstraction_learn_actions_tf.main(
                    dim, num_hidden_states, lr, 500, 100, False, 100, 1.0, 1.0, False, "0"
                )

                if np.any(np.isnan(best_accuracy)):
                    best_accuracy = 0.0

                results[num_hidden_states, lr, dim, run_idx] = best_accuracy

                with open(results_path, "wb") as file:
                    pickle.dump(results, file)
