"""Rules extracted from trees in trees.trees."""
from __future__ import annotations

import copy
from typing import List, Union, Tuple, Optional

import numpy as numpy
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import Tree

from planes.planes import APHyperplane, OHyperplane


class Premise:
    """A generic premise for a rule."""
    pass


class APPremise(APHyperplane, Premise):
    """An axis-parallel hyperplane of the form:
        fi in [ai, bi]
    that defines a continuous subspace on dimension `fi` delimited by a lower bound `ai` (included)
    and an upper bound `bi` (excluded).
    """
    def __init__(self, feat: int, low: float = -numpy.inf, upp: float = +numpy.inf):
        """

        Args:
            feat: The feature on which the hyperplane is defined
            low: The lower bound of the hyperplane. For premises of the form X <= a, `low` is `-numpy.inf`.
                    Defaults to `-numpy.inf`.
            upp: The upper bound of the hyperplane. For premises of the form X > a, `low` is `+numpy.inf`.
                    Defaults to `+numpy.inf`.
        """
        super().__init__(feat, low, upp)

    def __eq__(self, other):
        return isinstance(other, APPremise) and self.axis == other.axis and self.lower_bound == other.lower_bound and\
                self.upper_bound == other.upper_bound

    def __copy__(self):
        return APPremise(self.axis, self.lower_bound, self.upper_bound)

    def __deepcopy__(self, memodict={}):
        return APPremise(self.axis, self.lower_bound, self.upper_bound)

    def __call__(self, *args, **kwargs):
        x = args[0]

        return self.lower_bound <= x[self.axis] < self.upper_bound

    def tuple(self) -> Tuple[int, float, float]:
        """Return this premise as a (feature, lower bound, upper bound) tuple"""
        return self.axis, self.lower_bound, self.upper_bound


class OPremise(OHyperplane, Premise):
    """Oblique premise of the form:
        a1f1 + a2f2 + ... + amfm <= b
    where a1, ..., am, and b are scalars. Alternatively, formulated as ax <= b.
    """
    def __init__(self, coefficients: Union[List[float], numpy.ndarray], bound: float):
        """An oblique (multivariate) premise.

        Args:
            coefficients: Coefficients a1, ..., am
            bound: Bound scalar b
        """
        super().__init__(coefficients, bound)

    def __eq__(self, other):
        return isinstance(other, OPremise) and (self.coefficients == other.coefficients).all() and \
               self.bound == other.bound

    def __copy__(self):
        return OPremise(copy.copy(self.coefficients), self.bound)

    def __deepcopy__(self, memodict={}):
        return OPremise(copy.deepcopy(self.coefficients), self.bound)


class Rule:
    """Basic empty rule."""
    def __init__(self):
        self.head = None
        self.body = list()
        self.names = list()

    def __call__(self, *args, **kwargs):
        pass


class APRule(Rule):
    """Axis-parallel rule of the form
        head :- body
    where `head` is the outcome/consequence/label of the rule, and body is a set of premises of the form:
        fi in [ai, bi),
        ...
        fj in [aj, bj).

    Attributes:
        head (int): Head of the rule
        body (OPremise): Body of the rule
    """
    def __init__(self, head: int, body: List[APPremise]):
        """Axis-parallel rule.

        Args:
            head: Rule head
            body: Rule body
        """
        super().__init__()
        self.head = head
        # premises as tuples, need to construct APPremises
        self.body = {premise.axis: premise for premise in body}
        self.features = sorted(list(self.body.keys()))

    def __hash__(self):
        return sum([hash(premise) for premise in self.body]) + hash(self.head)

    def __eq__(self, other):
        if not isinstance(other, APRule):
            return False
        if self.features != other.features:
            return False
        return all([self[feat] == other[feat] for feat in self.features])

    def __len__(self):
        return len(self.body)

    def __getitem__(self, item):
        if item not in self.body:
            raise ValueError(f"Wrong item: expected one of ({list(self.body.keys())}), {item} found")
        return self.body[item]

    def __setitem__(self, k, v):
        if k == v.axis:
            self.body[k] = v
            self.features = sorted(self.body.keys())
        else:
            raise ValueError(f"Expected premise with feature {k}, {v.axis} found")

    def __delitem__(self, key):
        if key in self.body:
            del self.body[key]
            del self.features[self.features.index(key)]

    def __iter__(self):
        for feature in self.features:
            yield self.body[feature]

    def __call__(self, *args, **kwargs) -> bool | numpy.array:
        """Check whether the given array is within the premise, i.e., whether the premise covers the array or not.

        Args:
            *args:
            **kwargs:

        Returns:
            True if the element satisfies the premise, False otherwise.
        """
        x = args[0]
        if not isinstance(x, numpy.ndarray):
            raise ValueError(f"Not a numpy.ndarray: {type(x)}")

        if self.features[-1] > x.shape[0]:
            raise ValueError(f"Expected size at least {self.features[-1]}, {x.shape[0]} found")

        if x.ndim == 1:
            return all([premise(x) for premise in self.body.values()])
        else:
            return numpy.array([self(a) for a in x])

    def __copy__(self):
        # shallow copying: premises are references
        return APRule(self.head, [copy.copy(premise) for premise in self.body.values()])

    def __deepcopy__(self, memodict={}):
        # deep copying: completely new premises
        return APRule(self.head, [copy.deepcopy(premise) for premise in self.body.values()])

    def __str__(self):
        s = "Head: {0}\nBody:\n".format(self.head)
        for feat in self.features:
            s += f"\t{feat} in [{self.body[feat].lower_bound}, {self.body[feat].upper_bound})\n"
        return s

    def __contains__(self, item):
        return item in self.body

    def __add__(self, other):
        if not isinstance(other, APRule) and not isinstance(other, APPremise):
            raise ValueError(f"APRule or APPremise expected, {type(other)} found.")
        if isinstance(other, APRule):
            if len(set(self.features) & set(other.features)) > 0:
                raise ValueError("Can only add rules with premises on different axes.")
            body = [copy.deepcopy(premise) for premise in other.body.values()] +\
                   [copy.deepcopy(premise) for premise in self.body.values()]


            return APRule(self.head, body)

    def tuple(self) -> List[Tuple[int, float, float]]:
        """Return this rule as a list of (feature, lower bound, upper bound) tuples"""
        return [premise.tuple() for premise in self]


class ORule(Rule):
    """Oblique rule of the form
        head :- body
    where `head` is the outcome/consequence/label of the rule, and body is a list of premises of the form:
        a1f1 + a2f2 + ... + amfm <= b
    where a1, ..., am, and b are scalars. Alternatively, formulated as ax <= b.
    """
    def __init__(self, head: Optional[int], premises: List[OPremise] | List[OHyperplane]):
        super().__init__()
        self.head = head
        self.body = premises

    def __hash__(self):
        return hash(self.body) + hash(self.head)

    def __eq__(self, other):
        if not isinstance(other, ORule):
            return False

        return self.head == other.head and all([any([premise == other_premise
                                                     for other_premise in other.body])
                                                for premise in self.body])

    def __add__(self, other: Union[ORule, List[OPremise]]) -> ORule:
        """Create a new ORule with premises from both this and `other`."""
        if not (isinstance(other, ORule) or isinstance(other, OPremise) or isinstance(other, OHyperplane)):
            raise ValueError(f"Expected ORule or OPremise, {type(other)} found")
        if isinstance(other, Rule) and other.head != self.head:
            raise ValueError(f"Can't sum ORules with different heads: {self.head} and {other.head}")

        if isinstance(other, ORule):
            return ORule(self.head, self.body + other.body)
        elif isinstance(other, list):
            return ORule(self.head, self.body + other)

    def __sub__(self, other: OPremise) -> ORule:
        if not isinstance(other, OPremise):
            raise ValueError("Expected ORule or OPremise, {type(other)} found")
        if other not in self.body:
            return copy.deepcopy(self)

        new_body = copy.deepcopy(self.body)
        del new_body[new_body.index(other)]

        return ORule(self.head, new_body)

    def __len__(self):
        return len(self.body)

    def __getitem__(self, item):
        if item >= len(self):
            raise ValueError(f"Wrong item: expected one of [{0, len(self) - 1}), ..,  {item}] found")
        return self.body[item]

    def __setitem__(self, k, v):
        if k >= len(self):
            raise ValueError(f"Wrong item: expected one of [0, ..,  {len(self) - 1}], {k} found")
        self.body[k] = v

    def __iter__(self):
        for premise in self.body:
            yield premise

    def __call__(self, *args, **kwargs) -> bool | numpy.array:
        """Check whether the given array is within the premise, i.e., whether the premise covers the array or not.

        Args:
            *args:
            **kwargs:

        Returns:
            True if the element satisfies the premise, False otherwise.
        """
        x = args[0]
        if not isinstance(x, numpy.ndarray):
            raise ValueError("Not a numpy.ndarray: {0}".format(str(type(x))))

        if x.ndim == 1:
            try:
                return all([premise(x) for premise in self.body])
            except ValueError as e:
                raise e
        else:
            return numpy.array([self(x_) for x_ in x])

    def __copy__(self):
        # shallow copying: premises are references
        return ORule(self.head, [copy.copy(premise) for premise in self.body])

    def __deepcopy__(self, memodict={}):
        # deep copying: completely new premises
        return ORule(self.head, [copy.deepcopy(premise) for premise in self.body])

    def __str__(self):
        return "\n".join([str(premise) for premise in self.body])

    @staticmethod
    def from_aprule(aprule: APRule, dimensionality: int = 1) -> List[ORule]:
        """Create an ORule from an APRule

        Args:
            aprule: The rule to transform
            dimensionality: Dimensionality of the desired ORule

        Returns:
            An ORule with unitary coefficients for features appearing in the rule, and null coefficients for other
            features. The bound is given by a sum of the thresholds of the axis-parallel premises
        """
        appremises = aprule.tuple()
        if dimensionality < max([feat for feat, _, _ in appremises]):
            raise ValueError("Expected dimensionality >= {0}, {1} found".format(max([feat
                                                                                     for feat, _, _ in appremises]),
                                                                                dimensionality))

        # For positive premises (fi <= a) we only need one oblique hyperplane, and one coefficient:
        #   1 * fi <= a
        # The same holds for negative premises (fi >= b), only we need a negative coefficient of -1 to flip the
        # direction of the inequality.
        # -1 * fi <= b
        # For bounded premises we need two planes, e.g.
        #   fi in [a, b)
        # requires
        #   1 * ai <= a,  -1 * ai <= b
        # hence we introduce additional coefficients and bounds
        positive_coefficients = numpy.zeros(dimensionality,)
        negative_coefficients = numpy.zeros(dimensionality,)
        positive_obound = 0
        negative_obound = 0  # negative bound's sign must be flipped to account for future inequality flipping!
        for feat, low, upp in appremises:
            if not numpy.isinf(low) and not numpy.isinf(upp):
                positive_coefficients[feat] = 1
                negative_coefficients[feat] = -1
                positive_obound += upp
                negative_obound += low
            # fi >= ai
            elif not numpy.isinf(low) and numpy.isinf(upp):
                negative_coefficients[feat] = -1
                negative_obound += low
            # fi < ai
            elif numpy.isinf(low) and not numpy.isinf(upp):
                positive_coefficients[feat] = 1
                positive_obound += upp
        # need to negate the negative bound to flip the sign
        negative_obound = -negative_obound

        double_coefficients = numpy.argwhere(positive_coefficients * negative_coefficients < 0).squeeze()
        # each double hyperplane doubles the number of total planes
        nr_double_hyperplanes = 2 ** double_coefficients.size if double_coefficients.size > 0 else 0
        orules = list()
        # only unbounded premises, one orule is enough to define it
        if nr_double_hyperplanes == 0:
            if positive_coefficients.sum() != 0:
                orules.append(ORule(aprule.head, [OPremise(positive_coefficients, positive_obound)]))
            elif negative_coefficients.sum() != 0:
                orules.append(ORule(aprule.head, [OPremise(negative_coefficients, negative_obound)]))
        else:
            # create base (positive only) hyperplane
            positive_base = ORule(aprule.head, [OPremise(positive_coefficients, positive_obound)])
            # now add the negative planes to complete the base
            negative_base = ORule(aprule.head, [OPremise(negative_coefficients, negative_obound)])
            # full base w/o double coefficients
            base = positive_base + negative_base
            # copy the base
            base_copies = [copy.deepcopy(base) for _ in range(nr_double_hyperplanes)]
            # assign a combination of positive/negative coefficients for each double coefficient
            coefficients_to_assign = numpy.repeat([+1, -1], nr_double_hyperplanes / 2)
            coefficients_indices = [double_coefficients[i // 2] for i in range(len(double_coefficients) * 2)]
            for hyperplane, coefficient, coefficient_index in\
                    zip(range(nr_double_hyperplanes), coefficients_to_assign, coefficients_indices):
                base_copies[hyperplane][coefficient_index] = coefficient

        return orules


def from_cart(cart: DecisionTreeClassifier) -> List[APRule]:
    """Extract rules from the features of a sklearn.tree.DecisionTreeClassifier.
    Adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    Args:
        cart: The Decision Tree whose rules to extract
    Returns:
        The list of axis-parallel rules encoded by the `cart` Decision Tree
    """
    tree_paths = __all_paths(cart.tree_)
    tree_paths = list(filter(lambda path: len(path) > 1, tree_paths))
    features = [list(map(lambda node: cart.tree_.feature[abs(node)], path[:-1])) for path in tree_paths]
    features = [item for sublist in features for item in sublist]  # flatten list of lists
    thresholds = [list(map(lambda node: cart.tree_.threshold[abs(node)], path[:-1])) for path in tree_paths]
    leaves = [i for i in range(cart.tree_.node_count) if cart.tree_.children_left[i] == cart.tree_.children_right[i]]
    labels = {leaf: (cart.tree_.value[leaf][0]).argmax() for leaf in leaves}

    cart_rules = list()
    for feature_list, thresholds_list, path in zip(features, thresholds, tree_paths):
        if abs(path[-1]) not in leaves:
            continue

        rule_premises = {}
        # rule_features = features
        rule_features = [abs(cart.tree_.feature[p]) for p in path[:-1]]
        rule_label = labels[abs(path[-1])]

        # thresholds_ = thresholds[:-1]
        indices_per_feature = {feature: numpy.argwhere(rule_features == feature).flatten() for feature in rule_features}
        directions_per_feature = {f: [numpy.sign(path[k + 1]) for k in indices_per_feature[f] if k < len(path) - 1]
                                  for f in rule_features}

        for feature in rule_features:
            if len(indices_per_feature[feature]) == 1:
                threshold = thresholds[indices_per_feature[feature][0]][0]
                rule_premises[feature] = (-numpy.inf, threshold) if directions_per_feature[feature][0] < 0\
                                                                    else (threshold, numpy.inf)
            else:
                lower_bounds_idx = [index for index, direction in zip(indices_per_feature[feature],
                                                                      directions_per_feature[feature]) if direction > 0]
                upper_bounds_idx = [index for index, direction in zip(indices_per_feature[feature],
                                                                      directions_per_feature[feature]) if direction < 0]
                lower_bounds, upper_bounds = (numpy.array([thresholds[lower_idx] for lower_idx in lower_bounds_idx]),
                                              numpy.array([thresholds[upper_idx] for upper_idx in upper_bounds_idx]))

                if lower_bounds.shape[0] > 0 and upper_bounds.shape[0] > 0:
                    rule_premises[feature] = (max(lower_bounds), min(upper_bounds))
                elif lower_bounds.shape[0] == 0:
                    rule_premises[feature] = (-numpy.inf, min(upper_bounds).item())
                elif upper_bounds.shape[0] == 0:
                    rule_premises[feature] = (max(lower_bounds).item(), +numpy.inf)

        rule_premises = [APPremise(feat, low, upp) for (feat, (low, upp)) in rule_premises.items()]
        cart_rules.append(APRule(rule_label, rule_premises))

    return cart_rules


def __all_paths(tree: Tree) -> Union[List[Tuple], List[List[int]]]:
    """Retrieve all the possible paths in @tree.

    Arguments:
        tree: The decision tree internals.

    Returns:
        A list of lists of indices:[path_1, path_2, .., path_m] where path_i = [node_1, node_l].
    """
    paths = [[0]]
    left_child = tree.children_left[0]
    right_child = tree.children_right[0]

    if tree.capacity == 1:
        return paths

    paths = paths + \
            __rec_all_paths(tree, right_child, [0], +1) + \
            __rec_all_paths(tree, left_child, [0], -1)
    paths = sorted(set(map(tuple, paths)), key=lambda p: len(p))

    return paths


def __rec_all_paths(tree: Tree, node: int, path: List, direction: int):
    """Recursive call for the @all_paths function.

    Arguments:
        tree: The decision tree internals.
        node: The node whose path to expand.
        path: The path root-> `node`.
        direction:  +1 for right child, -1 for left child. Used to store the actual traversal.

    Returns:
        The enriched path.
    """
    # Leaf
    if tree.children_left[node] == tree.children_right[node]:
        return [path + [node * direction]]
    else:
        path_ = [path + [node * direction]]
        l_child = tree.children_left[node]
        r_child = tree.children_right[node]

        return path_ + \
               __rec_all_paths(tree, r_child, path_[0], +1) + \
               __rec_all_paths(tree, l_child, path_[0], -1)
