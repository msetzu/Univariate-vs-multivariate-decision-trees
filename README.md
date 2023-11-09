# Univariate and Multivariate Decision Trees

Univariate decision trees (e.g., CART, C4.5) have been the dominant model in decision tree induction, far outclassing multivariate trees despite their higher expressive power.
We analyze performance differences between univariate (e.g., CART, C4.5, C5) and multivariate (e.g., OC1, Omnivariate) decision trees by studying their behavior on both synthetic and a wide array of benchmark datasets used in the literature.

We aim to answer one questions: are datasets biased to favor one family over the other?

# Quickstart
Install dependencies through `pyenv` or `virtualenvwrapper`:
```shell
mkvirtualenv -p python3.11 trees_analysis

pip install -r src/requirements.txt
```
An additional set of notebook-only requirements can be installed through
```shell
pip install -r notebooks/requirements.txt
```

## Datasets
We analyze a wide array of benchmark datasets, which are available on [huggingface.co/mstz](https://huggingface.co/mstz).


## Experiments, notebooks and source
Source (`src/`) provides classes for several Decision Trees, e.g., OC1 (`trees/oc1.OC1`).
`experiments` holds several scripts:
- `train_on_synthetic_datasets.py` to train Trees on synthetic datasets,
- `train_on_benchmark_datasets.py` to train Trees on benchmark datasets.
Specifics on the two scripts available in the `experiments/README.md`.

You can find Jupyter notebooks in `notebooks/`:
- `Benchmark datasets.ipynb` contains analysis for benchmark datasets;
- `Synthetic datasets.ipynb` contains analysis for synthetic datasets datasets.

# Citation
This work is the implementation of the paper `Correlation and Unintended Biases on Univariate and Multivariate Decision Trees`, to appear in `IEEE Bigdata` 2023.
Citation key to appear soon.