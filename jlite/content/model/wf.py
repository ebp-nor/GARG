import networkx as nx
import numpy as np
from scipy.stats import binom


def wright_fisher(*, p0: float, n_ind: int, generations: int) -> list:
    """Simulate allele frequency trajectory"""
    x = np.repeat(p0, generations)
    for i in np.arange(1, generations):
        x[i] = binom(n_ind, x[i - 1]).rvs() / n_ind
    return x


# Functions related to networkx representation of Wright Fisher model
def sample_at_generation(G, *, generation, size, replace=True):
    i = np.min(
        G.subgraph([k for k, n in G.nodes(data=True) if n["generation"] == generation])
    )
    j = np.max(
        G.subgraph([k for k, n in G.nodes(data=True) if n["generation"] == generation])
    )
    k = np.random.choice(np.arange(i, j + 1), size, replace=replace)
    return np.array(sorted(k)), k


def set_layout(G, *, layout=None):
    """FIXME: Default: matrix with x left to right, y from top to bottom"""
    if layout is None:
        G.layout = {
            i: [n["individual"], G.generations - n["generation"] + 1]
            for i, n in G.nodes(data=True)
        }
    else:
        G.layout = layout
    return G


def generation_subgraph(G, *, gfrom=1, gto=None):
    """Generate subgraph view filtered on generation"""
    if gto is None:
        gto = G.generations

    def filter_node(n):
        retval = (G.nodes[n]["generation"] >= gfrom) and (
            G.nodes[n]["generation"] <= gto
        )
        return retval

    return nx.subgraph_view(G, filter_node=filter_node)


def wright_fisher_pop(
    *,
    n: int,
    generations: int,
    p0=None,
    init_gen: int = 1,
    s: int = 0,
    mu: float = 0.0,
    major: str = "a",
    minor: str = "A",
):
    """Model Wright Fisher population"""
    G = nx.Graph()
    G.generations = generations
    G.popsize = n
    z = np.array(
        [(x, y) for y in np.arange(1, generations + 1) for x in np.arange(1, n + 1)]
    )
    node_attr = list(
        {"individual": x[0], "generation": x[1], "x": x[0], "y": generations - x[1] + 1}
        for x in z
    )
    nodes = list(zip(np.arange(1, n * generations + 1), node_attr))
    G.add_nodes_from(nodes, allele=major, parent=None, tangled_parent=None)
    if p0 is not None:
        if isinstance(p0, int):
            sample_size = p0
        else:
            sample_size = p0 * n
        k, ktangled = sample_at_generation(
            G, generation=init_gen, size=sample_size, replace=False
        )
        for ii in k:
            G.node[ii]["allele"] = minor
    if (s == 0) and (mu == 0.0):
        # No selection, mutation
        parents = []
        tparents = []
        for gen in np.arange(1, generations + 1):
            k, kt = sample_at_generation(G, generation=gen, size=n)
            parents.extend(k)
            tparents.extend(kt)
        nx.set_node_attributes(
            G, dict(zip(np.arange(1, len(parents) + 1), parents)), "parent"
        )
        nx.set_node_attributes(
            G, dict(zip(np.arange(1, len(parents) + 1), tparents)), "tangled_parent"
        )

        nfrom = generation_subgraph(G, gto=G.generations - 1).nodes(data=True)
        nto = generation_subgraph(G, gfrom=2).nodes(data=True)
        nfrom = [g["parent"] for i, g in nfrom]
        nto = [i for i, g in nto]
        edges = [(x, y) for x, y in zip(nfrom, nto)]
        G.add_edges_from(edges)
    else:
        # selection, possibly mutation
        pass
    G = set_layout(G)
    return G
