import itertools
import os
import json
import sys

from datasets import load_dataset

from tqdm import tqdm
import fire
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pandas


__CWD = os.getcwd() + "/"
__SRC = __CWD + "../src/"
__DATA = __CWD + "../data/"
__EXPERIMENTS = __CWD + "../experiments/"
__NOTEBOOKS = __CWD

sys.path.append(__SRC)

pandas.set_option('display.max_rows', None)


def compute_coefficients(name, config, d, labels, pairs, batch):
    results = list()
    for feature_1, feature_2 in tqdm(pairs):
        data = d[[feature_1, feature_2]]
        logistic_regressor = LogisticRegression()
        logistic_regressor.fit(data, labels)

        results.append((name, config, logistic_regressor.coef_[0][0], logistic_regressor.coef_[0][1]))
    
    with open(f"{name}.{config}.slopes.batch:{batch}.json", "w") as log:
        json.dump(results, log)
    
    return results


def extract(name, config, batch, batch_size):
    dataset = load_dataset(name, config)["train"].to_pandas()

    scaler = StandardScaler()
    numeric_columns = [f for f in dataset.columns[:-1]
                       if dataset.dtypes[f].name.startswith("int")
                           or dataset.dtypes[f].name.startswith("float")
                           or dataset.dtypes[f].name.startswith("bool")]
    labels = dataset[dataset.columns[-1]]
    dataset = dataset[numeric_columns + [dataset.columns[-1]]]
    
    # select non-boolean columns
    non_boolean_columns = [i for i in dataset.columns[:-1] if set(dataset[i]) != {0, 1}]
    if len(non_boolean_columns) > 0:
        dataset.loc[:, non_boolean_columns] = scaler.fit_transform(dataset.loc[:, non_boolean_columns].values)

    feature_pairs = list(itertools.combinations(numeric_columns, 2))
    assigned_batch = feature_pairs[batch * batch_size: (batch + 1) * batch_size]

    compute_coefficients(name, config, dataset, labels, assigned_batch, batch)



if __name__ == "__main__":
    fire.Fire()
