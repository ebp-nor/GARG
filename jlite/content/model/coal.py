import random

import numpy as np
import scipy


class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"(x: {self.x}, y: {self.y})"

    def __add__(self, other):
        return Coordinate(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Coordinate(self.x - other.x, self.y - other.y)

    def __mul__(self, a):
        return Coordinate(a * self.x, a * self.y)

    def __truediv__(self, a):
        return Coordinate(self.x * 1 / a, self.y * 1 / a)


class Tree:
    def __init__(self, nodes):
        self.nodes = nodes
        self.mrca = self.nodes[-1]

    @property
    def segregating_sites(self):
        S = 0
        for n in self.nodes:
            S += n.mutations
        return S

    @property
    def total_branch_length(self):
        L = 0.0
        for n in self.nodes:
            L = L + n.tau
        return L

    @property
    def tmrca(self):
        return self.mrca.time

    @property
    def tcoal(self) -> list[float]:
        x = list()
        for i, n in enumerate(self.nodes):
            if n.left is None:
                continue
            x.append(n.time - self.nodes[i - 1].time)
        return x

    def __repr__(self):
        return f"Tree(nodes={self.nodes})"


class Node:
    def __init__(
        self,
        id: int,
        *,
        label=None,
        ancestor=None,
        left=None,
        right=None,
        mutations: int = 0,
        tau: float = 0.0,
    ):
        self.id = id
        if label is None:
            self.label = id
        self.ancestor = ancestor
        self.left = left
        self.right = right
        self.mutations = mutations
        self.tau = tau
        self._plot_coords = None

    @property
    def time(self):
        time = 0.0
        child = self.left
        while True:
            if child is None:
                break
            time = time + child.tau
            child = child.left
        return time

    @property
    def coords(self):
        """Return coordinates for plotting"""
        left = self.left
        if left is None:
            x = float(self.id)
        else:
            right = self.right
            x = (left.coords.x + right.coords.x) / 2
        return Coordinate(x=x, y=self.time)

    @property
    def plot_coords(self):
        if self._plot_coords is None:
            self.plot_coords = self.coords
        return self._plot_coords

    @plot_coords.setter
    def plot_coords(self, value):
        self._plot_coords = value

    @property
    def isleaf(self):
        return (self.left is None) and (self.right is None)

    def __repr__(self):
        aid = self.ancestor.id if self.ancestor is not None else None
        lid = self.left.id if self.left is not None else None
        rid = self.right.id if self.right is not None else None
        return (
            f"Node(id={self.id}, label={self.label}, "
            f"time={self.time}, tau={self.tau}, mutations={self.mutations} "
            f"ancestor_id={aid}, left={lid}, right={rid})"
        )


def add_mutations(tree, mutations):
    for i, _ in enumerate(tree.nodes):
        n = tree.nodes[i]
        if n == tree.mrca:
            continue
        tree.nodes[i].mutations = mutations[n.label]


def make_tree(ancestors, branches):
    """Convert ancestors and branches list to tree structure"""
    if isinstance(ancestors, list):
        ancestors = np.array(ancestors)
    if isinstance(branches, list):
        branches = np.array(branches)
    tree = Tree([Node(i) for i in range(len(ancestors) + 1)])
    for i in range(len(ancestors)):
        children = np.where(ancestors == i)[0]
        if len(children) > 0:
            left, right = children
            tree.nodes[i].left = tree.nodes[left]
            tree.nodes[i].right = tree.nodes[right]
        tree.nodes[i].ancestor = tree.nodes[ancestors[i]]
        tree.nodes[i].tau = branches[i]

    root = len(ancestors)
    left, right = np.where(ancestors == root)[0]
    tree.nodes[root].left = tree.nodes[left]
    tree.nodes[root].right = tree.nodes[right]
    # Relabel leaves according to postorder as they implicitly give x
    # coordinates
    order = postorder(ancestors)
    j = 0
    for i in order:
        if tree.nodes[i].isleaf:
            tree.nodes[i].id = j
            j = j + 1
    return tree


def postorder(x, index=None):
    """Postorder traverse list x by index"""
    order = list()
    if index is None:
        index = len(x)

    def _postorder(i):
        children = np.where(np.array(x) == i)
        if len(children[0]) == 0:
            order.append(i)
            return
        left, right = children[0]
        _postorder(left)
        _postorder(right)
        order.append(i)

    _postorder(index)
    return order


def sim_ancestry(samples):
    nodes = list(range(2 * samples - 1))
    ancestors = list(range(2 * samples - 2))
    branches = list(range(2 * samples - 2))
    # Initialize the age of all nodes to 0
    age = [0] * (2 * samples - 1)
    uncoalesced = list(range(samples))
    i = samples
    current_time = 0
    while i > 1:
        lmbda = i * (i - 1) / 2
        t = scipy.stats.expon(scale=1 / lmbda).rvs()
        current_time = current_time + t
        parent = max(uncoalesced) + 1
        age[parent] = current_time
        child1, child2 = random.sample(uncoalesced, 2)
        ancestors[child1] = parent
        ancestors[child2] = parent
        uncoalesced.remove(child1)
        uncoalesced.remove(child2)
        uncoalesced.append(parent)
        i = i - 1
    # Calculate branch lengths from node j as the age af the parent
    for j in nodes[0 : len(ancestors)]:
        branches[j] = age[ancestors[j]] - age[j]
    return ancestors, branches


def sim_mutations(branches, *, theta):
    tau = np.array(branches)
    Ttot = np.sum(tau)
    p = tau / Ttot
    S = scipy.stats.poisson(theta * Ttot / 1).rvs()
    mutations = scipy.stats.multinomial(n=S, p=p).rvs().flatten()
    return mutations
