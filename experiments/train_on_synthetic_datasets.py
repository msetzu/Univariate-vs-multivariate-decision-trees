import itertools
import json
import os
import pickle
import sys
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.append("../src/")

import fire
import numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from generators.correlation.twovariate import RSGenerator
from trees.multilinear import MultiLinearDT
from trees.structure.trees import Leaf

# random state
master_seed = 42
numpy.random.seed(master_seed)

nr_runs = 10
candidate_seeds = numpy.array(list(itertools.permutations(range(nr_runs ** 2), 2)))
# random seeds for each generator
seeds = candidate_seeds[numpy.random.choice(numpy.arange(candidate_seeds.shape[0]),
                                            nr_runs)]

hyperparameters = {
    "soft_margins": [0.001, 0.000001],
    "max_depth": [1, 16],
    "correlation": [i / 10 for i in range(11)],
    "noise": [0., 0.05, 0.10, 0.15, 0.20, 0.25],
    "slope": [
        (numpy.tan(numpy.radians(90)), 90),
        (numpy.tan(numpy.radians(45)), 45),
        (numpy.tan(numpy.radians(30)), 30),
        (numpy.tan(numpy.radians(15)), 15),
    ],
    "run": list(range(nr_runs)),
}


def validate_tree(ground_labels, predicted_labels):
    report = classification_report(ground_labels, predicted_labels,
                                   output_dict=True,
                                   zero_division=0)["weighted avg"]
    res = {k: v for k, v in report.items() if k in ["precision", "recall", "f1-score"]}
    res["accuracy"] = accuracy_score(ground_labels, predicted_labels)
    res["roc_auc"] = roc_auc_score(ground_labels, predicted_labels)
    res["average_precision"] = average_precision_score(ground_labels, predicted_labels)

    return res


def _split_data(x, y, random_state):
    return train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)


def train_multivariate_tree(x_tr, x_ts, y_tr, y_ts, soft_margin, max_depth):
    model = MultiLinearDT()
    model = model.fit(x_tr, y_tr.astype(int),
                      min_eps=10e-6,
                      max_depth=max_depth if max_depth != 1 else 2,
                      node_hyperparameters={"C": soft_margin})

    nr_nodes = len(model.nodes)
    max_nr_nonzero_coefficients = x_tr.shape[1]
    nodes_coefficients = numpy.array([sum(node.hyperplane.coefficients != 0) / max_nr_nonzero_coefficients
                                      for node in model.nodes.values() if not isinstance(node, Leaf)])
    mean_nonzero_coefficients = nodes_coefficients.mean()
    std_nonzero_coefficients = nodes_coefficients.std()

    validation = {
        "training": validate_tree(y_tr.astype(int), model.predict(x_tr).round().astype(int)),
        "test": validate_tree(y_ts.astype(int), model.predict(x_ts).round().astype(int)),
        "tree": {
            "size": nr_nodes,
            "mean_node_size": mean_nonzero_coefficients,
            "std_node_size": std_nonzero_coefficients,
        }
    }

    return model, validation


def train_univariate_tree(x_tr, x_ts, y_tr, y_ts, max_depth):
    model = DecisionTreeClassifier(max_depth=max_depth)
    model = model.fit(x_tr, y_tr)

    nr_nodes = model.tree_.node_count
    max_nr_nonzero_coefficients = x_tr.shape[1]
    nodes_coefficients = numpy.array([1 / max_nr_nonzero_coefficients for _ in range(nr_nodes)])
    mean_nonzero_coefficients = nodes_coefficients.mean()
    std_nonzero_coefficients = nodes_coefficients.std()

    validation = {
        "training": validate_tree(y_tr.astype(int), model.predict(x_tr).round().astype(int)),
        "test": validate_tree(y_ts.astype(int), model.predict(x_ts).round().astype(int)),
        "tree": {
            "size": nr_nodes,
            "mean_node_size": mean_nonzero_coefficients,
            "std_node_size": std_nonzero_coefficients,
        }
    }

    return model, validation


def train_multivariate_on(correlation, noise, slope, degrees_slope, max_depth, soft_margin, run, out):
    # randomly select random seeds
    base_name = f"{out}/correlation:{correlation}_slope:{degrees_slope}_noise:{noise}_maxdepth:{max_depth}_softmargin:{soft_margin}_run:{run}_type:multivariate"

    if not os.path.exists(f"{base_name}.validation.json"):
        random_state = seeds[run][0]
        random_state_optional = seeds[run][1]
        generator = RSGenerator(random_state=random_state,
                                 random_state_optional=random_state_optional)
        cfg_dictionary = {
            "split": "multivariate",
            "soft_margin": soft_margin,
            "max_depth": int(max_depth),
            "correlation": correlation,
            "slope": slope,
            "degrees_slope": degrees_slope,
            "noise": noise,
            "random_state": random_state,
            "random_state_optional": random_state_optional,
            "run": run
        }    

        dataset, flipped = generator.generate(correlation=correlation,
                                              slope=slope,
                                              label_noise=noise)
        data, labels = dataset[:, :-1], dataset[:, -1].squeeze()
        data_tr, data_ts, labels_tr, labels_ts = _split_data(data, labels,
                                                             random_state=random_state)

        start_time = time.time()
        model, validation = train_multivariate_tree(data_tr, data_ts, labels_tr, labels_ts,
                                                    soft_margin=soft_margin, max_depth=max_depth)
        end_time = time.time()
        validation["configuration"] = cfg_dictionary
        validation["elapsed_time"] = end_time - start_time
        validation["tree_structure"] = model.json()

        with open(f"{base_name}.validation.json", "w") as log:
            json.dump(validation, log)

        with open(f"{base_name}.data.npy", "wb") as log:
            numpy.save(log, dataset)

        with open(f"{base_name}.data.flipped.npy", "wb") as log:
            numpy.save(log, flipped)


def train_univariate_on(correlation, noise, slope, degrees_slope, max_depth, soft_margin, run, out):
    # randomly select random seeds
    base_name = f"{out}/correlation:{correlation}_slope:{degrees_slope}_noise:{noise}_maxdepth:{max_depth}_softmargin:{soft_margin}_run:{run}_type:univariate"

    if not os.path.exists(f"{base_name}.validation.json"):
        random_state = seeds[run][0]
        random_state_optional = seeds[run][1]
        generator = RSGenerator(random_state=random_state,
                                 random_state_optional=random_state_optional)
        cfg_dictionary = {
            "split": "univariate",
            "soft_margin": soft_margin,
            "max_depth": int(max_depth),
            "correlation": correlation,
            "slope": slope,
            "degrees_slope": degrees_slope,
            "noise": noise,
            "random_state": random_state,
            "random_state_optional": random_state_optional,
            "run": run
        }
        generator = RSGenerator(random_state=seeds[run][0],
                                 random_state_optional=seeds[run][1])
        dataset, flipped = generator.generate(correlation=correlation,
                                              slope=slope,
                                              label_noise=noise)
        data, labels = dataset[:, :-1], dataset[:, -1].squeeze()
        data_tr, data_ts, labels_tr, labels_ts = _split_data(data, labels,
                                                             random_state=random_state)

        start_time = time.time()
        model, validation = train_univariate_tree(data_tr, data_ts, labels_tr, labels_ts,
                                                  max_depth=max_depth)
        end_time = time.time()
        validation["configuration"] = cfg_dictionary
        validation["elapsed_time"] = end_time - start_time

        with open(f"{base_name}.validation.json", "w") as log:
            json.dump(validation, log)

        with open(f"{base_name}.tree.pickle", "wb") as log:
            pickle.dump(model, log)

        with open(f"{base_name}.data.npy", "wb") as log:
            numpy.save(log, dataset)

        with open(f"{base_name}.data.flipped.npy", "wb") as log:
            numpy.save(log, flipped)


def multivariatetree(n_jobs: int = -1, out: str = "../data/experiments/multivariate"):
    n_threads = n_jobs if n_jobs > 1 else multiprocessing.cpu_count()

    hyperparameter_configurations = itertools.product(*hyperparameters.values())
    hyperparameter_configurations = list(hyperparameter_configurations)

    total_configs = len(hyperparameter_configurations)

    if not os.path.exists(out):
        os.mkdir(out)

    print(f"Running {total_configs} models...")
    with ProcessPoolExecutor(max_workers=n_threads - 1) as pool:
        for soft_margin, max_depth, correlation, noise, (slope, degrees_slope), run in hyperparameter_configurations:
            pool.submit(train_multivariate_on, correlation, noise, slope, degrees_slope, max_depth, soft_margin, run, out)


def univariatetree(n_jobs: int = -1, out: str = "../data/synthetic/"):
    n_threads = n_jobs if n_jobs > 1 else multiprocessing.cpu_count() - 1

    hyperparameter_configurations = itertools.product(*hyperparameters.values())
    hyperparameter_configurations = list(hyperparameter_configurations)
    total_configs = len(hyperparameter_configurations)

    if not os.path.exists(out):
        os.mkdir(out)

    print(f"Running {total_configs} models...")
    with ProcessPoolExecutor(max_workers=n_threads - 1) as pool:
        for soft_margin, max_depth, correlation, noise, (slope, degrees_slope), run in hyperparameter_configurations:
            pool.submit(train_univariate_on, correlation, noise, slope, degrees_slope, max_depth, 1., run, out)


if __name__ == '__main__':
    fire.Fire()
