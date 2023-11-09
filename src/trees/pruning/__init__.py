from abc import abstractmethod
from typing import List
import copy

from trees.splits.evaluation import label_deviation, gini
from trees.structure.trees import ObliqueTree, InternalNode


class Gardener:
    def __init__(self):
        pass

    @abstractmethod
    def prune(self, tree: ObliqueTree, **kwargs) -> ObliqueTree:
        """
        Prune `tree` post-hoc.

        Args:
            tree: The tree to prune.

        Return:
            A pruned version of the input tree.
        """
        pass

class PostGardener(Gardener):
    def __init__(self):
        super(self, PostGardener).__init__()

    def prunee(self, tree: ObliqueTree, **kwargs) -> ObliqueTree:
        max_depth = kwargs.get("max_depth", -1)
        validation_data = kwargs["data"]
        validation_labels = kwargs["labels"]

        pruned_tree = copy.deepcopy(tree)

        return

    def bottom_up_prune(self, node: InternalNode, direction: str = "down",
                        tree: ObliqueTree, validation_data: numpy.ndarray, validation_labels: numpy.ndarray,
                        classes: numpy.ndarray):
        """Prune this tree in a bottom-up fashion. When `direction` is "down" reach the one-to-last layer and try to replace it, while when `direction` is up traverse
        the tree back up.

        Args:
            node: The current node
            ancestors: Node ancestors
            direction: The current tree traversal direction
            linear_model: The linear model to use
        """
    def should_prune(node: InternalNode) -> bool:
        node_data, node_labels = node._data, node._label
        # compute linear model std
        current_node_impurity = gini(node.hyperplane, validation_data, validation_labels, classes)
        parent_node_impurity = gini(tree.nodes[tree.parent[node.node_id]].hyperplane,
                                    validation_data, validation_labels, classes)

        return parent_node_impurity <= current_node_impurity

    if direction == "down":
        # down direction: looking for the last layer to fit nodes
        left_child, right_child = node.children
        # last layer
        if isinstance(left_child, Leaf) and isinstance(right_child, Leaf):
            if should_prune(node, tree, validation_data, validation_labels, classes):
                # construct new leaves
                probabilities = numpy.bincount(node._label) / node._label.size
                children = [Leaf(probabilities), Leaf(1 - probabilities)]
                tree.nodes[tree.parent[node.node_id]].children = children
                tree.parent[children[0]] = tree.parent[node.node_id]
                tree.parent[children[1]] = tree.parent[node.node_id]

                # go back up the tree
                self.bottom_up_prune(ancestors[-1], "up", tree, validation_data, validation_labels, classes)
        elif isinstance(left_child, Leaf) and isinstance(right_child, InternalNode):
            self.bottom_up_prune(node.children[1],
                                 "down",
                                 tree, validation_data, validation_labels, classes)
        elif isinstance(left_child, InternalNode) and isinstance(right_child, Leaf):
            self.bottom_up_prune(node.children[0],
                                 "down",
                                 tree, validation_data, validation_labels, classes)
        else:
            # both internal nodes, iterate
            self.bottom_up_prune(node.children[0],
                                 "down",
                                 tree, validation_data, validation_labels, classes)
            self.bottom_up_prune(node.children[1],
                                 "down",
                                 tree, validation_data, validation_labels, classes)

    else:
        # going back up
        if should_prune(node, tree, validation_data, validation_labels, classes):
            # construct new leaves
            probabilities = numpy.bincount(node._label) / node._label.size
            children = [Leaf(probabilities), Leaf(1 - probabilities)]
            tree.nodes[tree.parent[node.node_id]].children = children
            tree.parent[children[0]] = tree.parent[node.node_id]
            tree.parent[children[1]] = tree.parent[node.node_id]

            # go back up the tree
            if node.node_id > 1:
                self.bottom_up_prune(tree.nodes[tree.parent[node.node_id]],
                                     "up",
                                     tree, validation_data, validation_labels, classes)
