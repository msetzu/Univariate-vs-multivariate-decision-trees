from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Tuple, Iterable, Sequence, TypeVar, Set

import numpy

from planes.planes import OHyperplane, APHyperplane

T = TypeVar("T")


class Node(ABC):
    """Node (internal or leaf) of a Decision Tree."""
    def __init__(self, hyperplane: Optional[OHyperplane] = None):
        self.hyperplane = hyperplane
        self._data = None
        self._label = None

    @abstractmethod
    def __eq__(self, other):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __invert__(self):
        pass

    @abstractmethod
    def json(self):
        pass

    @staticmethod
    def from_json(json_obj) -> List:
        if json_obj["type"] == "leaf":
            return Leaf.from_json(json_obj)
        else:
            return InternalNode.from_json(json_obj)


class InternalNode(Node):
    """Internal node of a Decision Tree."""
    def __init__(self, hyperplane: Optional[OHyperplane] = None):
        super().__init__(hyperplane)
        self.children = [None, None]

    def __hash__(self):
        return hash(self.hyperplane) + 1000

    def __eq__(self, other):
        return isinstance(other, InternalNode) and self.hyperplane == other.hyperplane

    def __call__(self, *args, **kwargs) -> numpy.ndarray:
        """Return a routing array."""
        return self.hyperplane(args[0])

    def __repr__(self):
        return repr(self.hyperplane)

    def __invert__(self):
        node = InternalNode(~self.hyperplane)
        # flipping children position because we flipped the hyperplane
        node.children = [copy.deepcopy(self.children[1]), copy.deepcopy(self.children[0])]

        return node

    def json(self) -> Dict:
        base = dict({"type": "node"})
        base.update({"hyperplane": self.hyperplane.json()})

        return base

    @staticmethod
    def from_json(json_obj):
        # OHyperplane
        if json_obj["hyperplane"]["type"] == "oblique":
            return InternalNode(OHyperplane.from_json(json_obj["hyperplane"]))
        # APHyperplane
        else:
            return InternalNode(APHyperplane.from_json(json_obj["hyperplane"]))

    def __deepcopy__(self, memodict={}):
        node = InternalNode(copy.deepcopy(self.hyperplane))
        node.children = [copy.deepcopy(self.children[0]), copy.deepcopy(self.children[1])]

        return node


class LinearInternalNode(InternalNode):
    def __init__(self, hyperplane: Optional[OHyperplane] = None):
        super().__init__(hyperplane)
        self.children = [None, None]
        self.companion_hyperplane = None

    def __eq__(self, other):
        return isinstance(other, LinearInternalNode) and \
            super().__eq__(other) and \
            self.companion_hyperplane == other.companion_hyperplane


class Leaf(Node):
    """Leaf of a Decision Tree.

    Attributes:
        _label: Probability of each label in this leaf.
    """
    def __init__(self, class_probability: numpy.ndarray):
        """Create a new Leaf with the given class probabilities.

        Args:
            class_probability: The class probabilities
        """
        super().__init__()
        self.label = class_probability
        self._data = None

    def __hash__(self):
        return hash(self.label) + 1000

    def __eq__(self, other):
        return isinstance(other, Leaf) and (self.label == other.label).all()

    def __call__(self, *args, **kwargs) -> numpy.ndarray:
        """Get the label probability.

        Args:
            *args:
            **kwargs:

        Returns:
            The label probability.
        """
        return self.label

    def __repr__(self):
        return repr(self.label.tolist())

    def __invert__(self):
        return Leaf(1 - self.label)

    def __deepcopy__(self, memodict={}):
        return Leaf(self.label.copy())

    def json(self) -> Dict:
        base = dict({"type": "leaf"})
        base.update({"label": self.label.tolist()})

        return base

    @staticmethod
    def from_json(json_obj):
        return Leaf(numpy.array(json_obj["label"]))


class Tree(ABC):
    """A Decision Tree"""
    def __init__(self):
        self.paths = list()
        self.path_labels = list()
        self.root = None

    def __eq__(self, other):
        if not type(other) is type(self) or self.root != other.root:
            return False
        return self.__rec__eq__(self.root.children[0], other.root.children[0]) and \
            self.__rec__eq__(self.root.children[1], other.root.children[1])

    @abstractmethod
    def __hash__(self):
        pass

    def __rec__eq__(self, node: Node, other_node: Node):
        if isinstance(node, Leaf) and not isinstance(other_node, Leaf) or \
                not isinstance(node, Leaf) and isinstance(other_node, Leaf):
            return False
        if isinstance(node, Leaf) and isinstance(other_node, Leaf):
            return node == other_node

        return self.__rec__eq__(node.children[0], other_node.children[0]) and \
            self.__rec__eq__(node.children[1], other_node.children[1])

    @abstractmethod
    def predict(self, data: numpy.ndarray) -> numpy.ndarray | float:
        """Predict the given data `x` by routing it along the tree."""
        pass

    @abstractmethod
    def coverage(self, data: numpy.ndarray) -> numpy.ndarray:
        """Compute the coverage matrix of the given `data`."""
        pass

    def depth_first_accumulate(self, node: InternalNode | Leaf, foo: callable, accumulated: List):
        """Apply `foo` to each node in this tree in a depth-first manner, accumulate its results

        Args:
            node: The current node to which apply `foo`
            foo: The function to apply
            accumulated: The accumulated results
        """
        if isinstance(node, Leaf):
            return accumulated
        else:
            accumulated.append(foo(node))
            accumulated += self.depth_first_accumulate(node.children[0], foo, accumulated)
            accumulated += self.depth_first_accumulate(node.children[1], foo, accumulated)

            return accumulated


class ObliqueTree(Tree):
    def __init__(self, root: Optional[InternalNode] = Node):
        super(ObliqueTree, self).__init__()
        self.root = root
        self.path_matrices = list()
        self.path_bounds = list()
        self.path_labels = list()

    def __hash__(self):
        hashes = [hash(p) for p in self.paths]

        return sum(hashes)

    def predict(self, data: numpy.ndarray, output_paths: bool = False) -> numpy.array | Tuple[numpy.array, List[Path]]:
        """Predict the given `data` by routing it along the tree. To get the paths of each prediction, set
        `output_paths` to `True`. Defaults to False."""
        coverage_matrix = self.coverage(data)
        # TODO: check this
        predicting_paths = (numpy.argwhere(x_coverage)[0][0] for x_coverage in coverage_matrix.transpose())
        predictions = numpy.array([self.path_labels[i].argmax() for i in predicting_paths])

        if output_paths:
            predicting_paths = (numpy.argwhere(x_coverage).item() for x_coverage in coverage_matrix.transpose())
            paths = [self.paths[i] for i in predicting_paths]

            return predictions, paths
        else:
            return predictions

    def coverage(self, data: numpy.ndarray) -> numpy.ndarray:
        """Compute the coverage matrix of the given `data`.

        Returns:
            A K x N binary matrix C where C[i, j] = True if self.paths[i] covers data[j], False otherwise.
        """
        n = data.shape[0]
        # repeat bounds: from bounds vector b obtain bounds matrix B = [b, b, ..., b] where b is repeated n times
        expanded_bounds = [numpy.repeat(bounds.reshape(-1, 1), n, axis=1) for bounds in self.path_bounds]
        # AX
        ax_projections = (numpy.dot(path, data.transpose()) for path in self.path_matrices)  # AX for all As
        # AX <= b
        coverage_vectors = [(projection <= bounds).all(axis=0)
                            for projection, bounds in zip(ax_projections, expanded_bounds)]  # AX <= B
        coverage_matrix = numpy.vstack(coverage_vectors)  # (l paths by n records) coverage matrix

        return coverage_matrix

    def build(self):
        """Build internal path representation, invoked after `fit`."""
        self._compile_node_infos()
        self._build_structure()
        self._build_paths()
        hyperplane_paths = [[node.hyperplane for node in path[:-1]] for path in self.paths]
        self.path_matrices = [numpy.vstack([h.coefficients for h in path]) for path in hyperplane_paths]
        self.path_bounds = [numpy.vstack([h.bound for h in path]).squeeze() for path in hyperplane_paths]
        self.path_labels = numpy.array([path[-1].label for path in self.paths], dtype="object")

    def _build_structure(self):
        self.children = dict()
        self.parent = dict()

        for node_id, node in self.nodes.items():
            if isinstance(node, Leaf):
                self.children[node_id] = list()
            else:
                self.children[node_id] = [child.node_id for child in node.children]
                for child in node.children:
                    self.parent[child.node_id] = node_id
        self.parent[1] = 1

        # build ancestors
        self.ancestors = {1: set()}
        for node_id in self.nodes:
            self.ancestors[node_id] = {1, self.parent[node_id]}
            current_ancestor_id = node_id
            while current_ancestor_id != 1:
                parent_id = self.parent[current_ancestor_id]
                self.ancestors[node_id].add(parent_id)
                current_ancestor_id = parent_id
        self.ancestors = {key: sorted(value) for key, value in self.ancestors.items()}

        # build descendants
        self.descendants = {1: set(self.nodes.keys())}
        for node_id in self.nodes:
            self.descendants[node_id] = self._descendants(node_id, current_descendants=list())
        self.descendants = {key: sorted(value)[1:] for key, value in self.descendants.items()}

    def _descendants(self, node_id: int, current_descendants: List) -> Set[int]:
        if isinstance(self.nodes[node_id], Leaf):
            return set(current_descendants + [node_id])
        return self._descendants(node_id * 2, current_descendants + [node_id]) | self._descendants(node_id * 2 + 1,
                                                                                                   current_descendants + [node_id])

    def _compile_node_infos(self):
        root = self.root
        root.node_id = 1

        self.nodes = {1: root}
        self.__rec_gather_nodes(root.children[0], 2)
        self.__rec_gather_nodes(root.children[1], 3)

    def __rec_gather_nodes(self, node: Node, index: int):
        self.nodes.update({index: node})
        node.node_id = index

        if isinstance(node, InternalNode):
            self.__rec_gather_nodes(node.children[0], index * 2)
            self.__rec_gather_nodes(node.children[1], index * 2 + 1)

    def _build_paths(self):
        """Construct the paths in this tree, starting from the root. These paths will then be used in prediction phase."""
        leaves_ids = [node_id for node_id, node in self.nodes.items() if isinstance(node, Leaf)]
        leaves_paths_ids = [list(self.ancestors[leaf]) + [leaf] for leaf in leaves_ids]
        leaves_paths = [[self.nodes[node_in_path] for node_in_path in path] for path in leaves_paths_ids]
        # by construction left children (node_id % 2 == 0) lie in the hyperplane, while right children don't,
        # hence need to invert the paths routing towards right children
        for path in leaves_paths:
            for i, node in enumerate(path):
                if i > 0 and node.node_id % 2 == 1:
                    # right child: need to negate the parent node
                    path[i - 1] = ~path[i - 1]
        leaves_paths = [ObliquePath(path) for path in leaves_paths]

        self.paths = leaves_paths
        self.paths_ids = leaves_paths_ids

    def json(self):
        tree_json = dict()
        tree_json["1"] = {"node": self.root.json(), "parent": None}
        tree_json.update(self.__rec_json(self.root.children[0], index=2, parent=1))
        tree_json.update(self.__rec_json(self.root.children[1], index=3, parent=1))

        return tree_json

    def __rec_json(self, node: Node, index: int, parent: int):
        tree_json = {str(index): {"node": node.json(), "parent": parent}}
        if isinstance(node, InternalNode):
            update_left = self.__rec_json(node.children[0], index * 2, index)
            update_right = self.__rec_json(node.children[1], index * 2 + 1, index)
            tree_json.update(update_left)
            tree_json.update(update_right)

        return tree_json

    @staticmethod
    def from_json(json_file: str | Dict) -> ObliqueTree:
        """Extract a tree from the given `json_file`.

        Args:
            json_file: Path to the JSON file encoding the tree.

        Returns:
            The tree encoded in the `json_file`
        """
        if isinstance(json_file, str):
            with open(json_file, "r") as log:
                json_obj = json.load(log)
        else:
            json_obj = json_file

        nodes = {int(k): Node.from_json(json_obj[k]["node"]) for k in json_obj.keys()}
        root = nodes[1]
        ObliqueTree.__rec_from_json(root, 1, nodes)
        tree = ObliqueTree(root)

        tree.build()

        return tree

    @staticmethod
    def __rec_from_json(node: InternalNode | Leaf, node_id: int, nodes: Dict):
        left_child, right_child = node_id * 2, node_id * 2 + 1
        if isinstance(node, InternalNode):
            # internal node
            if left_child in nodes:
                node.children[0] = nodes[left_child]
                ObliqueTree.__rec_from_json(node.children[0], left_child, nodes)
            if right_child in nodes:
                node.children[1] = nodes[right_child]
                ObliqueTree.__rec_from_json(node.children[1], right_child, nodes)
        else:
            return


class Path(Sequence[T]):
    """A path in the tree."""
    def __init__(self, path: Iterable):
        self.path = list(path)

    def __deepcopy__(self, memodict={}) -> Path:
        return type(self)([copy.deepcopy(node) for node in self.path])

    def __eq__(self, other):
        if isinstance(other, ObliquePath):
            if len(self) == len(other):
                return [this_node == other_node for this_node, other_node in zip(self.path, other.path)]
        return False

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        return self.path[item]

    def __iter__(self):
        for el in self.path:
            yield el

    def __setitem__(self, item, value):
        self.path[item] = value

    def coverage(self, data: numpy.ndarray) -> numpy.array:
        return self.__rec_coverage(data, self.path, numpy.full((data.shape[0]), True))

    def __rec_coverage(self, data: numpy.ndarray, remaining_path: List[Node], covered: numpy.array):
        node = remaining_path[0]
        if isinstance(node, Leaf):
            return covered

        node_coverage = node(data)  # node.__call__(data) computes the coverage of the given data
        covered[covered] = node_coverage

        return self.__rec_coverage(data[node_coverage], remaining_path[1:], covered)


class APPath(Path):
    """A path in an axis-parallel tree."""
    def __init__(self, path: Iterable):
        super().__init__(path)

    def __getitem__(self, item):
        return self.path[item]

    def __len__(self):
        return len(self.path)

    def json(self) -> List:
        json_obj = list()
        for node in self.path:
            if isinstance(node, Leaf):
                json_obj.append(node.label)
            else:
                json_obj.append([node.hyperplane.axis, node.hyperplane.lower_bound, node.hyperplane.upper_bound])

        return json_obj

    @staticmethod
    def from_json(json_obj: List) -> ObliquePath:
        path = list()
        for feat, low, upp in json_obj[:-1]:
            path.append(InternalNode(APHyperplane(feat, low, upp)))
        path.append(Leaf(numpy.array(json_obj[-1])))

        return ObliquePath(path)


class ObliquePath(Path):
    """A path in an ObliqueTree."""
    def __init__(self, path: Iterable):
        super().__init__(path)

    def __eq__(self, other):
        return isinstance(other, ObliquePath) and all([this_el == other_el for this_el, other_el in zip(self.path,
                                                                                                        other.path)])

    def __len__(self):
        return len(self.path)

    def __getitem__(self, item):
        return self.path[item]

    def __hash__(self):
        coefficients_hash = sum(sum([node.hyperplane.coefficients for node in self.path[:-1]]))
        bounds_hash = sum([node.hyperplane.bound for node in self.path[:-1]])
        label = self.path[-1].grounding_labels

        return int(coefficients_hash + bounds_hash + label.argmax().item() * 10000)

    def __repr__(self):
        return " - ". join([repr(node) for node in self.path])

    def json(self) -> List:
        json_obj = list()
        for node in self.path:
            if isinstance(node, Leaf):
                json_obj.append(node.label)
            else:
                json_obj.append([node.hyperplane.coefficients.tolist(), node.hyperplane.bound])

        return json_obj

    @staticmethod
    def from_json(json_obj: List) -> ObliquePath:
        path = list()
        for coefficients, bound in json_obj[:-1]:
            path.append(InternalNode(OHyperplane(numpy.array(coefficients), bound)))
        path.append(Leaf(numpy.array(json_obj[-1])))

        return ObliquePath(path)
