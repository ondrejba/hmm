import itertools
import pickle
import click
import numpy as np


@click.command()
@click.argument("path")
def main(path):

    with open(path, "rb") as file:
        results = pickle.load(file)

    key = list(results.keys())[0]
    key_values = [[] for _ in range(len(key))]

    for key in results.keys():
        for idx, item in enumerate(key):
            key_values[idx].append(item)

    key_values = [list(sorted(set(x))) for x in key_values]
    num_runs = np.max(key_values[-1])

    total = 0

    for setting in itertools.product(*key_values[:-1]):
        accuracies = []
        for run_idx in range(num_runs):
            tmp_key = (*setting, run_idx)

            if tmp_key in results:
                accuracies.append(results[tmp_key])
                total += 1

        print("{}: {:.2f}% +- {:.2f}".format(setting, np.mean(accuracies) * 100, np.std(accuracies) * 100))

    print("finished {:d} runs".format(total))


if __name__ == "__main__":
    main()
