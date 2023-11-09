import itertools
import os
import sys

from datasets import load_dataset

from tqdm import tqdm
import fire
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import pandas
import numpy

pandas.set_option("display.max_rows", None)

__CWD = os.getcwd() + "/"
__SRC = __CWD + "../src/"
__DATA = __CWD + "../data/"

sys.path.append(__SRC)



def compute_coefficients(name, config, d, labels, pairs, batch, batch_size):
    results = list()
    base = batch_size * batch
    for i, (feature_1, feature_2) in tqdm(enumerate(pairs), total=len(pairs)):
        data = d[[feature_1, feature_2]]
        logistic_regressor = LogisticRegression()
        logistic_regressor.fit(data, labels)

        slope = -logistic_regressor.coef_[0][0] / logistic_regressor.coef_[0][1] if logistic_regressor.coef_[0][1] != 0 else numpy.nan

        results.append((name, config, base + i, logistic_regressor.coef_[0][0], logistic_regressor.coef_[0][1], slope))
    
    if len(results) > 0:
        pandas.DataFrame(results).to_csv(f"{name}.{config}.slopes.batch:{batch}.csv", index=False, header=None)
    
    return results


def extract(name, config, batch, batch_size):
    if os.path.exists(f"{name}.{config}.slopes.batch:{batch}.csv"):
        return

    dataset = load_dataset("mstz/" + name, config)["train"].to_pandas()
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
    assigned_batch = feature_pairs[int(batch) * int(batch_size): (int(batch) + 1) * int(batch_size)]

    compute_coefficients(name, config, dataset, labels, assigned_batch, int(batch), int(batch_size))


if __name__ == "__main__":
    fire.Fire()
