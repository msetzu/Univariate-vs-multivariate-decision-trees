import itertools
import json
import os
import pickle
import sys
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple

import fire

import numpy
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset

sys.path.append("../src/")

from trees.multilinear import MultiLinearDT
from trees.structure.trees import Leaf, ObliqueTree

random_state = 42
numpy.random.seed(random_state)
datasets_names = [
    ("mstz/acute_inflammation", "inflammation"),
    ("mstz/adult", "income"),
    ("mstz/arcene", "arcene"),
    ("mstz/arhythmia", "has_arhythmia"),
    ("mstz/australian_credit", "australian_credit"),
    ("mstz/balance_scale", "is_balanced"),
    ("mstz/bank", "subscription"),
    ("mstz/blood", "blood"),
    ("mstz/breast", "cancer"),
    ("mstz/car", "car_binary"),
    ("mstz/contraceptive", "contraceptive"),
    ("mstz/compas", "two-years-recidividity"),
    ("mstz/covertype", "covertype_0"),
    ("mstz/dexter", "dexter"),
    ("mstz/electricity", "electricity"),
    ("mstz/fertility", "fertility"),
    ("mstz/german", "loan"),
    ("mstz/gisette", "gisette"),
    ("mstz/glass", "vehicles"),
    ("mstz/haberman", "haberman"),
    ("mstz/heart_failure", "death"),
    ("mstz/heloc", "risk"),
    ("mstz/higgs", "higgs"),
    ("mstz/hill", "hill"),
    ("mstz/hypo", "has_hypo"),
    ("mstz/ipums", "ipums"),
    # ("mstz/isolet", "isolet"),
    # ("mstz/liver", "liver"),
    ("mstz/lrs", "lrs_0"),
    ("mstz/magic", "magic"),
    ("mstz/madelon", "madelon"),
    ("mstz/house16", "house16"),
    ("mstz/ionosphere", "ionosphere"),
    # ("mstz/liver", "liver"),
    ("mstz/magic", "magic"),
    ("mstz/musk", "musk"),
    ("mstz/nbfi", "default"),
    ("mstz/ozone", "8hr"),
    # ("mstz/p53", "p53"),
    ("mstz/page_blocks", "page_blocks_binary"),
    ("mstz/phoneme", "phoneme"),
    ("mstz/pima", "pima"),
    ("mstz/pol", "pol"),
    ("mstz/pums", "pums"),
    ("mstz/planning", "planning"),
    ("mstz/post_operative", "post_operative_binary"),
    ("mstz/seeds", "seeds_0"),
    ("mstz/seeds", "seeds_1"),
    ("mstz/seeds", "seeds_2"),
    ("mstz/segment", "brickface"),
    ("mstz/shuttle", "shuttle_0"),
    ("mstz/sonar", "sonar"),
    # ("mstz/soybean", "diaporthe_stem_canker"),
    ("mstz/spambase", "spambase"),
    ("mstz/spect", "spect"),
    ("mstz/speeddating", "dating"),
    ("mstz/steel_plates", "steel_plates_0"),
    ("mstz/student_performance", "math"),
    ("mstz/sydt", "sydt"),
    ("mstz/toxicity", "toxicity"),
    ("mstz/twonorm", "twonorm"),
    # ("mstz/uscensus", "uscensus"),
    ("mstz/vertebral_column", "abnormal"),
    ("mstz/wall_following", "wall_following_0"),
    ("mstz/waveformnoiseV1", "waveformnoiseV1_0"),
    ("mstz/wine_origin", "wine_origin_0"),
    ("mstz/wine", "wine"),
    # ("mstz/yeast", "yeast"),
]

hyperparameters = {
    "dataset": datasets_names,
    "soft_margins": [0.000001],
    "max_depth": [1, 8]
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


def _transform_data(dataset):
    numeric_columns = [f for f in dataset.columns[:-1] if dataset.dtypes[f].name.startswith("int")
                       or dataset.dtypes[f].name.startswith("float")
                       or dataset.dtypes[f].name.startswith("bool")]
    label_column = dataset.columns[-1]
    dataset[numeric_columns] = dataset[numeric_columns].astype("float")

    data, labels = dataset[numeric_columns].values, dataset[label_column].values.squeeze()
    # select non-boolean columns
    non_boolean_columns = [i for i in range(data.shape[1]) if set(data[:, i]) != {0, 1}]
    scaler = StandardScaler()
    data[:, non_boolean_columns] = scaler.fit_transform(data[:, non_boolean_columns])

    return data, labels, scaler


def _split_data(x, y, random_state):
    return train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)


def train_multivariate_tree(x_tr, x_ts, y_tr, y_ts, soft_margin, max_depth) -> Tuple[ObliqueTree, Dict]:
    model = MultiLinearDT()
    model = model.fit(x_tr, y_tr.astype(int),
                      min_eps=10e-6,
                      max_depth=max_depth,
                      node_hyperparameters={
                          "C": soft_margin
                      })

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


def train_univariate_tree(x_tr, x_ts, y_tr, y_ts, max_depth)-> Tuple[DecisionTreeClassifier, Dict]:
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


def train_multivariate_on(name, config, soft_margin,  max_depth, out):
    name_only = name.split("/")[1]
    base_name = f"{out}/benchmark_{name_only}_{config}_{max_depth}_{soft_margin}_multivariate"

    if not os.path.exists(f"{base_name}.validation.json"):
        cfg_dictionary = {
            "split": "multivariate",
            "soft_margin": soft_margin,
            "max_depth": int(max_depth),
            "name": name_only,
            "config": config
        }

        dataset = load_dataset(name, config, split="train").to_pandas()
        data, labels, scaler = _transform_data(dataset)
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

        with open(f"{base_name}.scaler.pickle", "wb") as log:
            pickle.dump(scaler, log)


def train_univariate_on(name, config, soft_margin,  max_depth, out):
    name_only = name.split("/")[1]
    base_name = f"{out}/benchmark_{name_only}_{config}_{max_depth}_{soft_margin}_univariate"

    if not os.path.exists(f"{base_name}.validation.json"):
        cfg_dictionary = {
            "split": "univariate",
            "soft_margin": soft_margin,
            "max_depth": int(max_depth),
            "name": name_only,
            "config": config
        }
        dataset = load_dataset(name, config, split="train").to_pandas()
        data, labels, scaler = _transform_data(dataset)
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

        with open(f"{base_name}.scaler.pickle", "wb") as log:
            pickle.dump(scaler, log)


def multivariatetree(n_jobs: int = -1, out: str = "../data/benchmark/"):
    n_threads = n_jobs if n_jobs > 1 else multiprocessing.cpu_count()

    hyperparameter_configurations = itertools.product(*hyperparameters.values())
    hyperparameter_configurations = list(hyperparameter_configurations)

    total_configs = len(hyperparameter_configurations)

    if not os.path.exists(out):
        os.mkdir(out)

    print(f"Running {total_configs} models...")
    with ProcessPoolExecutor(max_workers=n_threads - 1) as pool:
        for (name, config), soft_margin, max_depth in hyperparameter_configurations:
            pool.submit(train_multivariate_on, name, config, soft_margin, max_depth, out)


def univariatetree(n_jobs: int = -1, out: str = "../data/benchmark/"):
    n_threads = n_jobs if n_jobs > 1 else multiprocessing.cpu_count() - 1

    hyperparameter_configurations = itertools.product(*hyperparameters.values())
    hyperparameter_configurations = list(hyperparameter_configurations)
    total_configs = len(hyperparameter_configurations)

    if not os.path.exists(out):
        os.mkdir(out)

    print(f"Running {total_configs} models...")
    with ProcessPoolExecutor(max_workers=n_threads - 1) as pool:
        for (name, config), soft_margin, max_depth in hyperparameter_configurations:
            pool.submit(train_univariate_on, name, config, 1., max_depth, out)


if __name__ == '__main__':
    fire.Fire()
