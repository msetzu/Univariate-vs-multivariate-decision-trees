from __future__ import annotations

import logging
import os
from typing import Optional, Dict

import numpy
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression, Lasso, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from planes.planes import OHyperplane
from rules.rules import from_cart, ORule
from trees.splits.evaluation import gini
from trees.structure.trees import Node, Leaf, InternalNode, ObliqueTree


class MultiLinearDT(ObliqueTree):
    """Parametric Oblique Tree built using the provided model family for internal node split."""

    def __init__(self, root: Optional[InternalNode] = None):
        super(MultiLinearDT, self).__init__(root)

    def fit(self, data: numpy.ndarray, labels: numpy.ndarray, max_depth: int = numpy.inf,
            min_eps: float = 0.00000001, min_samples: int = 10, node_hyperparameters: Optional[Dict] = None):
        """Learn an Oblique Tree whose internal nodes use a `strategy` split function.

        Args:
            data: The training set
            labels: The training set labels
            max_depth: Maximum depth of the Decision Tree
            min_eps: Minimum improve in the learning metric to keep on creating new nodes
            node_hyperparameters: Optional training hyperparameters to train the internal nodes.

        Returns:
            This GeometricDT, fit to the given `data` and `labels`.
        """
        # create root
        logging.debug("Fitting tree with:")
        logging.debug(f"\tmax depth: {max_depth}")
        logging.debug(f"\tmin eps: {min_eps}")

        os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"

        logging.debug("Fitting root...")
        classes = numpy.unique(labels)
        self.node_hyperparameters = node_hyperparameters
        root = self.step(None, data, labels, classes, min_eps=min_eps, min_samples=min_samples,
                         node_hyperparameters={} if node_hyperparameters is None else node_hyperparameters)
        self.root = root

        left_child_indices = root(data)
        left_child_data = data[left_child_indices]
        left_child_labels = labels[left_child_indices]
        right_child_data = data[~left_child_indices]
        right_child_labels = labels[~left_child_indices]

        logging.debug("Fitting left child...")
        self.__rec_fit(root, left_child_data, left_child_labels, classes, "left",
                       depth=1, max_depth=max_depth, min_eps=min_eps,
                       min_samples=min_samples, node_hyperparameters=node_hyperparameters)
        logging.debug("Fitting right child...")
        self.__rec_fit(root, right_child_data, right_child_labels, classes, "right",
                       depth=1, max_depth=max_depth, min_eps=min_eps,
                       min_samples=min_samples, node_hyperparameters=node_hyperparameters)

        self.build()

        return self

    def step(self, parent_node: Optional[InternalNode], data: numpy.ndarray, labels: numpy.ndarray,
             classes: numpy.ndarray, min_eps: float = 0.01, min_samples: int = 100,
             node_hyperparameters: Optional[Dict] = None) -> Node:
        """Compute a learning step, i.e., learn a hyperplane for the given `data` and `labels`, with parent node
        `parent_node`.

        Args:
            data: The data routed to the node to learn
            labels: The labels of the data
            classes: The set of classes of the whole dataset
            parent_node: The subtree to fit
            min_eps: Minimum improve in the learning metric to keep on creating new nodes.
            min_samples: Minimum number of samples in leaves.
            node_hyperparameters: Optional training hyperparameters to train the internal nodes.

        Returns:
            The learned separating hyperplane
        """
        if numpy.unique(labels).size == 1:
            probabilities = numpy.zeros((2,))
            probabilities[int(labels[0])] = 1.
            fit_node = Leaf(probabilities)

        else:
            dual = data.shape[0] < data.shape[1]
            models = [Ridge(),
                      ElasticNet(),
                      SGDClassifier(loss="hinge", penalty="l1" if not dual else "l2", max_iter=100000),
                      LinearRegression(),
                      LinearSVC(penalty="l1" if not dual else "l2", dual=dual, max_iter=100000, **node_hyperparameters),
                      Lasso(),
                      DecisionTreeClassifier(max_depth=1)]
            model_names = ["ridge", "elastic", "sgd-svm", "linear", "svm", "lasso", "tree"]
            ginis = list()
            hyperplanes = list()
            for model_name, model in zip(model_names, models):
                model.fit(data, labels)
                if model_name == "tree":
                    if model.tree_.node_count > 1:
                        extracted_rules = from_cart(model)
                        extracted_rules = [ORule.from_aprule(rule, dimensionality=data.shape[1])
                                           for rule in extracted_rules]
                        extracted_rule = [rule[0] for rule in extracted_rules][0][0]
                        hyperplane = OHyperplane(extracted_rule.coefficients, extracted_rule.bound)
                    else:
                        hyperplanes.append(numpy.zeros(data.shape[1],))
                else:
                    hyperplane = OHyperplane(model.coef_.transpose().squeeze(), model.intercept_)
                hyperplane.bound = hyperplane.bound if isinstance(hyperplane.bound, float) else hyperplane.bound[0]
                fit_node_gini = gini(hyperplane, data, labels, classes)
                ginis.append(fit_node_gini)
                hyperplanes.append(hyperplane)
            ginis = numpy.array(ginis)
            best_idx = numpy.argmin(ginis)
            fit_node_gini = ginis[best_idx]
            hyperplane = hyperplanes[best_idx]

            if parent_node is not None:
                parent_gini = gini(parent_node, data, labels, classes)
                if parent_gini - fit_node_gini < min_eps:
                    # error stopping criterion
                    logging.debug(f"Reached minimum error delta of {min_eps}")
                    probabilities = numpy.bincount(labels) / labels.size
                    if probabilities.size == 0:
                        probabilities = numpy.array([0.5, 0.5])
                    fit_node = Leaf(probabilities)
                else:
                    fit_node = InternalNode(hyperplane)
            else:
                fit_node = InternalNode(hyperplane)
            
            fit_node.model_name = model_names[best_idx]
            fit_node.ginis = ginis

        fit_node._data = data
        fit_node._label = labels

        return fit_node

    def __rec_fit(self, parent_node: Optional[Node], data: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray,
                  direction: str, depth: int = 0, max_depth: int = numpy.inf,
                  min_eps: float = 0.0001, min_samples: int = 100, node_hyperparameters: Optional[Dict] = None):
        """Recursively fit the Geometric Decision Tree rooted in `root`.

        Args:
            data: The training set
            labels: The training set labels
            classes: The training set classes
            parent_node: The subtree to fit

            depth: Depth of the root of the current subtree.
            max_depth: Maximum depth of the Decision Tree.
            min_eps: Minimum improve in the learning metric to keep on creating new nodes.
            min_samples: Minimum number of samples in leaves.
            node_hyperparameters: Optional training hyperparameters to train the internal nodes.


        Returns:
            This tree, fit to `data` and `labels`.
        """
        # trying to recurse on an already built tree
        if direction == "left" and parent_node.children[0] is not None \
                or direction == "right" and parent_node.children[1] is not None:
            return

        # maximum depth reached, create a leaf
        if depth >= max_depth - 1:
            logging.debug(f"Reached max depth of {max_depth}")
            probabilities = numpy.bincount(labels) / labels.size
            if probabilities.size == 0:
                probabilities = numpy.array([0.5, 0.5])
            elif probabilities.size == 1:
                probabilities = numpy.zeros(2,)
                probabilities[int(labels[0])] = 1.
            children = [Leaf(probabilities), Leaf(1 - probabilities)]
            parent_node.children = children

        elif data.shape[0] < min_samples:
            logging.debug(f"Reached minimum samples of {min_samples}")
            probabilities = numpy.bincount(labels) / labels.size
            if probabilities.size == 0:
                probabilities = numpy.array([0.5, 0.5])
            elif probabilities.size == 1:
                probabilities = numpy.zeros(2,)
                probabilities[int(labels[0])] = 1.
            child = Leaf(probabilities)
            child_index = 0 if direction == "left" else 1
            parent_node.children[child_index] = child

        else:
            step_result = self.step(parent_node, data, labels, classes,
                                    min_eps=min_eps, min_samples=min_samples, node_hyperparameters=node_hyperparameters)

            if isinstance(step_result, Leaf):
                if direction == "left":
                    parent_node.children[0] = step_result
                else:
                    parent_node.children[1] = step_result
            else:
                fit_node = step_result
                child_index = 0 if direction == "left" else 1
                parent_node.children[child_index] = fit_node

                # recurse on children
                data_indices = fit_node.hyperplane(data)
                left_child_data, left_child_labels = data[data_indices], labels[data_indices]
                right_child_data, right_child_labels = data[~data_indices], labels[~data_indices]

                logging.debug(f"Fitting child on {left_child_data.shape[0]} nodes...")
                self.__rec_fit(fit_node, left_child_data, left_child_labels, classes, "left",
                               depth=depth + 1, max_depth=max_depth, min_eps=min_eps, min_samples=min_samples,
                               node_hyperparameters=node_hyperparameters)
                logging.debug(f"Fitting child on {right_child_data.shape[0]} nodes...")
                self.__rec_fit(fit_node, right_child_data, right_child_labels, classes, "right",
                               depth=depth + 1, max_depth=max_depth, min_eps=min_eps, min_samples=min_samples,
                               node_hyperparameters=node_hyperparameters)

