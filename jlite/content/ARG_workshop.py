import collections
import itertools
import json
import os
import sys
import subprocess
import tempfile
from datetime import datetime

import msprime
import networkx as nx
import numpy as np
import pandas as pd
import tqdm
import tskit
import yaml
import zarr
from IPython.core.display import HTML
from jupyterquiz import display_quiz
from matplotlib import collections as mc
from matplotlib import pyplot as plt


path = os.path.dirname(os.path.normpath(__file__))


def load_quiz(section):
    with open(os.path.join(path, "quiz.yaml"), encoding="utf-8") as fh:
        try:
            quiz = yaml.safe_load(fh)
        except yaml.YAMLError as e:
            print(e)
    if section in quiz.keys():
        return quiz[section]
    return


scilife_colors = {
    "lime": "#a7c947",
    "lime25": "#e9f2d1",
    "lime50": "#d3e4a3",
    "lime75": "#bdd775",
    "teal": "#045c64",
    "teal25": "#c0d6d8",
    "teal50": "#82aeb2",
    "teal75": "#43858b",
    "aqua": "#4c979f",
    "aqua25": "#d2e5e7",
    "aqua50": "#a6cbcf",
    "aqua75": "#79b1b7",
    "grape": "#491f53",
    "grape25": "#d2c7d4",
    "grape50": "#a48fa9",
    "grape75": "#77577e",
    "lightgray": "#e5e5e5",
    "mediumgray": "#a6a6a6",
    "darkgray": "#3f3f3f",
    "black": "#202020",
}

color_dict = {
    "--jq-multiple-choice-bg": scilife_colors[
        "lime"
    ],  # Background for the question part of multiple-choice questions
    "--jq-mc-button-bg": scilife_colors[
        "teal25"
    ],  # Background for the buttons when not pressed
    "--jq-mc-button-border": scilife_colors["teal50"],  # Border of the buttons
    "--jq-mc-button-inset-shadow": scilife_colors[
        "teal75"
    ],  # Color of inset shadow for pressed buttons
    "--jq-many-choice-bg": scilife_colors[
        "lime"
    ],  # Background for question part of many-choice questions
    "--jq-numeric-bg": scilife_colors[
        "lime"
    ],  # Background for question part of numeric questions
    "--jq-numeric-input-bg": scilife_colors[
        "lime75"
    ],  # Background for input area of numeric questions
    "--jq-numeric-input-label": scilife_colors[
        "lime"
    ],  # Color for input of numeric questions
    "--jq-numeric-input-shadow": scilife_colors[
        "lime75"
    ],  # Color for shadow of input area of numeric questions when selected
    "--jq-incorrect-color": scilife_colors["grape"],  # Color for incorrect answers
    "--jq-correct-color": scilife_colors["teal50"],  # Color for correct answers
    "--jq-text-color": scilife_colors["lime25"],  # Color for question text
}


class DownloadProgressBar(tqdm.tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


class Workbook:
    styles = open(os.path.join(path, "styles/custom.css"), encoding="utf-8").read()
    css = f"<style>{styles}</style>"
    js = "<script src='https://d3js.org/d3.v7.min.js'></script>"
    # See https://github.com/jupyterlite/jupyterlite/issues/407#issuecomment-1353088447
    html_text = [
        """<table style="width: 100%;"><tr>
        <td style="text-align: left;">✅ Your notebook is ready to go!</td>""",  # => 0
        """<td style="text-align: right;">
        <button type="button" id="button_for_indexeddb">Clear JupyterLite local storage
        </button></td>""",  # => 1 (omit if not in jlite)
        "</tr></table>",  # => 2
        """<script>
        window.button_for_indexeddb.onclick = function(e) {
            window.indexedDB.open('JupyterLite Storage').onsuccess = function(e) {
                // There are also other tables that I'm not clearing:
                // "counters", "settings", "local-storage-detect-blob-support"
                let tables = ["checkpoints", "files"];

                let db = e.target.result;
                let t = db.transaction(tables, "readwrite");

                function clearTable(tablename) {
                    let st = t.objectStore(tablename);
                    st.count().onsuccess = function(e) {
                        console.log("Deleting " + e.target.result +
                        " entries from " + tablename + "...");
                        st.clear().onsuccess = function(e) {
                            console.log(tablename + " is cleared!");
                        }
                    }
                }

                for (let tablename of tables) {
                    clearTable(tablename);
                }
            }
        };
        </script>""",  # => 3 (omit if not in jlite)
    ]

    # Used for making SVG formatting smaller
    small_class = "x-lab-sml"
    small_style = (
        ".x-lab-sml .sym {transform:scale(0.6)} "
        ".x-lab-sml .lab {font-size:7pt;}"  # All labels small
        ".x-lab-sml .x-axis .tick .lab {"
        "font-weight:normal;transform:rotate(90deg);text-anchor:start;dominant-baseline:central;}"  # noqa
    )

    def __init__(self):
        name = type(self).__name__
        self.quiz = load_quiz(name)

    def question(self, label):
        display_quiz(self.quiz[label], colors=color_dict)

    @staticmethod
    def download(url):
        return DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        )

    @property
    def setup(self):
        if "pyodide" in sys.modules:
            return HTML(self.js + self.css + "".join(self.html_text))
        else:
            return HTML(
                self.js + self.css + "".join([self.html_text[0], self.html_text[2]])
            )


class HOWTO(Workbook):
    def __init__(self):
        super().__init__()
        self.quiz["day"][0]["answers"][0]["value"] = datetime.today().day


class Workbook1A(Workbook):
    def __init__(self):
        super().__init__()


class Workbook1B(Workbook):
    def __init__(self):
        super().__init__()


class Workbook1C(Workbook):
    def __init__(self):
        super().__init__()


class Workbook1D(Workbook):
    def __init__(self):
        super().__init__()
        # The Fst quiz gives different answers between jlite and others, so we need to adjust it here
        pop_sizes = [3e4, 2e4, 3e4, 1e4, 10e4]  # pick some variable sizes for each population 
        model = msprime.Demography.stepping_stone_model(pop_sizes, migration_rate=0.001, boundaries=True)
        # sample the same number from each pop
        samples = {'pop_0': 6, 'pop_1': 6, 'pop_2': 6, 'pop_3': 6, 'pop_4': 6}
        base_ts = msprime.sim_ancestry(samples, sequence_length=1e6, demography=model, recombination_rate=1e-8, random_seed=1)
        ts = msprime.sim_mutations(base_ts, rate=1e-8, random_seed=1)
        ans1 = ts.Fst([ts.samples(population=0), ts.samples(population=3)])
        ans2 = ts.Fst([ts.samples(population=0), ts.samples(population=3)], mode="branch")
        self.quiz["Fst"][0]["answers"][0]["value"] = f"{ans1:.4f}"
        self.quiz["Fst"][1]["answers"][0]["value"] = f"{ans2:.4f}"

class Workbook1E(Workbook):
    def __init__(self):
        super().__init__()


class Workbook1F(Workbook):
    def __init__(self):
        super().__init__()


class Workbook2A(Workbook):
    def __init__(self):
        super().__init__()


class Workbook2B(Workbook):

    def __init__(self):
        super().__init__()

    @staticmethod
    def run_psmc(params):
        from psmc_python.model import PSMC

        data_file = params[0]
        data = np.load(data_file)
        n_iter = params[1] if len(params) > 1 else 10
        start_window = params[2] if len(params) > 2 else 0
        end_window = params[3] if len(params) > 3 else data.shape[1]
        data = data[:, start_window: end_window]
        print(f"Using {data.shape[1]} 100bp windows for PSMC inference of {data_file}", flush=True)
        theta0 = np.sum(data) / (data.shape[0] * data.shape[1])
        rho0 = theta0 / 5

        psmc_model = PSMC(t_max=15, n_steps=64, pattern='1*4+25*2+1*4+1*6', progress_bar=None)
        psmc_model.param_recalculate()

        initial_params = [theta0, rho0, 15] + [1.] * (psmc_model.n_free_params - 3)
        bounds = [(1e-4, 1e-1), (1e-5, 1e-1), (12, 20)] + [(0.1, 10)] * (psmc_model.n_free_params - 3)

        name = data_file.replace(".npy", "")
        loss_list, params_history = psmc_model.EM(initial_params, bounds, x=data, n_iter=n_iter, name=name)
        psmc_model.save_params(name + ".json")
        # Only choose e.g. 100_000 x 100bp windows. You can change this (or omit it) to include
        # more of the genome in order to increase accuracy, but it will take longer to run


class Workbook2C(Workbook):
    def __init__(self):
        super().__init__()


class Workbook2D(Workbook):
    def __init__(self):
        super().__init__()


class FwdWrightFisherSimulator:
    def __init__(self, population_size, seq_len=1000, random_seed=8):
        self.flags = tskit.NODE_IS_SAMPLE
        self.rng = np.random.default_rng(seed=random_seed)
        self.tables = tskit.TableCollection(sequence_length=seq_len)
        self.tables.time_units = "generations"
        self.current_population = self.initialize(population_size)

    def run(self, gens, simplify=True, **kwargs):
        # NB: set time of current_population=0 & count downwards (generations are negative).
        for neg_gens in -np.arange(gens):
            self.current_population = self.reproduce(
                self.current_population, neg_gens - 1
            )

        # On output, rebase the times so the current generation is at time 0
        self.tables.nodes.time += gens
        # Reorder the nodes in time order, youngest first
        time_order = np.argsort(self.tables.nodes.time, kind="stable")
        node_map = {u: i for i, u in enumerate(time_order)}
        self.tables.subset(time_order)
        for k in self.current_population.keys():
            self.current_population[k] = [
                node_map[v] for v in self.current_population[k]
            ]
        self.tables.sort()  # Sort edges into canonical order, required for converting to a tree seq

        if simplify:
            if "samples" not in kwargs:
                kwargs["samples"] = [
                    u for nodes in self.current_population.values() for u in nodes
                ]
            node_map = self.tables.simplify(**kwargs)
            for k in self.current_population.keys():
                self.current_population[k] = [
                    node_map[v] for v in self.current_population[k]
                ]
        return self.tables.tree_sequence()

    def initialize(self, diploid_population_size):
        """
        Save a population to the tskit_tables and return a Python dictionary
        mapping the newly created individual ids to a pair of genomes (node ids)
        """
        temp_pop = {}  # make an empty dictionary
        for _ in range(diploid_population_size):
            # store in the TSKIT tables
            i = self.tables.individuals.add_row(parents=(tskit.NULL, tskit.NULL))
            maternal_node = self.tables.nodes.add_row(self.flags, time=0, individual=i)
            paternal_node = self.tables.nodes.add_row(self.flags, time=0, individual=i)
            # Add to the dictionary: map the individual ID to the two node IDs
            temp_pop[i] = [maternal_node, paternal_node]
        return temp_pop

    def reproduce(self, previous_pop, current_time):
        temp_pop = {}
        prev_individual_ids = list(previous_pop.keys())
        for _ in range(len(previous_pop)):
            mum, dad = self.rng.choice(prev_individual_ids, size=2, replace=False)
            i = self.tables.individuals.add_row(parents=(mum, dad))
            maternal_node = self.tables.nodes.add_row(time=current_time, individual=i)
            paternal_node = self.tables.nodes.add_row(time=current_time, individual=i)
            temp_pop[i] = (maternal_node, paternal_node)

            # Now add inheritance paths to the edges table, ignoring recombination
            self.add_edges(self.rng.permuted(previous_pop[mum]), maternal_node)
            self.add_edges(self.rng.permuted(previous_pop[dad]), paternal_node)
        return temp_pop

    def add_edges(self, randomly_ordered_parent_nodes, child_node):
        parent_node = randomly_ordered_parent_nodes[0]
        L = self.tables.sequence_length
        self.tables.edges.add_row(parent=parent_node, child=child_node, left=0, right=L)


class FwdWrightFisherRecombSim(FwdWrightFisherSimulator):
    def add_edges(self, randomly_ordered_parent_nodes, child_node):
        L = self.tables.sequence_length
        num_breakpoints = self.rec_rng.poisson(L * self.recombination_rate, size=1)
        breakpoint_positions = np.unique(
            [0, *self.rec_rng.integers(L, size=num_breakpoints), L]
        )
        choose_genome = 0
        for left, right in zip(breakpoint_positions[:-1], breakpoint_positions[1:]):
            self.tables.edges.add_row(
                left=left,
                right=right,
                parent=randomly_ordered_parent_nodes[choose_genome],
                child=child_node,
            )
            choose_genome = 1 if choose_genome == 0 else 0

    def __init__(
        self, population_size, seq_len=1000, recombination_rate=1e-8, random_seed=427
    ):
        self.recombination_rate = recombination_rate
        self.rec_rng = np.random.default_rng(seed=random_seed)
        super().__init__(population_size, seq_len, random_seed)


## Functions for drawing & plotting


def basic_genealogy_viz(
    tables_or_ts, ax=None, *, show_node_ids=True, show_individuals=None, title=None
):
    """
    Plot genealogical tables (designed for non-recombinant fwd sims). As this is designed to
    display tables as they are built, either a TreeSequence or a TableCollection can be passed in
    If show_individuals is False, we don't group nodes by individual, but reposition them
    to avoid overlap if possible. If show_individuals is None, we group but don't plot individual
    outlines. If show_individuals is True, we plot hexagonal outlines
    """
    try:
        tables = tables_or_ts.tables
    except AttributeError:
        tables = tables_or_ts
    if ax is None:
        ax = plt.gca()
    node_pos_x = {}
    individual_pos = collections.defaultdict(list)
    # assumes nodes are ordered by time
    order = {u: 0 for u in range(tables.nodes.num_rows)}
    individual_pos = collections.defaultdict(list)
    for time, nodes in itertools.groupby(enumerate(tables.nodes), lambda x: x[1].time):
        individual = None
        x_pos = 0
        node_dict = {}
        for i, (u, nd) in enumerate(nodes):
            parent = tables.edges.parent[tables.edges.child == u]
            if show_individuals is None or show_individuals:
                # order by individual ID
                node_dict[u] = nd.individual
            else:
                # order by parent node ID
                node_dict[u] = order[parent[0]] if len(parent) == 1 else i
        # sort by value
        node_dict = {
            k: node_dict[k]
            for k in sorted(node_dict.keys(), key=lambda x: node_dict[x])
        }
        for i, (u, v) in enumerate(node_dict.items()):
            if show_individuals is None or show_individuals:
                if v != individual:
                    x_pos += 1
                individual = v
                individual_pos[individual].append([x_pos, time])
            else:
                order[u] = i
            node_pos_x[u] = x_pos
            x_pos += 1
    has_edge = np.zeros(tables.nodes.num_rows, dtype=bool)
    has_edge[tables.edges.parent] = True
    colour_vec = np.where(has_edge, "tab:orange", "tab:blue")
    for plt_sample, marker in ((True, "s"), (False, "o")):
        use = (tables.nodes.flags & tskit.NODE_IS_SAMPLE) == 0
        if plt_sample:
            use = np.logical_not(use)
        x = np.array([node_pos_x[u] for u in range(tables.nodes.num_rows)])
        if np.any(use):
            ax.scatter(x[use], tables.nodes.time[use], c=colour_vec[use], marker=marker)
    tweak_y = 0
    if show_node_ids:
        tweak_y = 0.02 * tables.nodes.time.max()
        for i, y in enumerate(tables.nodes.time):
            ax.annotate(i, (node_pos_x[i], y - tweak_y), ha="center", va="top")
    if len(individual_pos) and show_individuals:
        individual_pos = np.array([np.mean(v, axis=0) for v in individual_pos.values()])
        ax.scatter(
            individual_pos[:, 0],
            individual_pos[:, 1] - tweak_y,
            facecolors="none",
            edgecolors="0.9",
            marker="H",
            s=1400,
        )
    for edge in tables.edges:
        x = node_pos_x[edge.child]
        dx = node_pos_x[edge.parent] - x
        ax.arrow(x, tables.nodes[edge.child].time, dx, 1, color="lightblue")
    ax.set_xticks([])
    max_x = tables.nodes.time.max() + tweak_y * 3
    ax.set_ylim(None, max_x)
    ax.set_yticks(np.arange(tables.nodes.time.max() + 1))
    ax.set_ylabel("Time ago (generations)")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if title is not None:
        ax.set_title(title)


rotated_label_style = (
    ".node > .lab {font-size: 80%}"
    ".leaf > .lab {text-anchor: start; transform: rotate(90deg) translate(6px)}"
)


def draw_pedigree(
    ped_ts, ax=None, title=None, show_axis=False, font_size=None, ts_edge=None, **kwargs
):
    # If ts_edge is None, do not overlay tskit edges, otherwise specify a color
    # or "black" if True
    if ts_edge is True:
        ts_edge = "black"
    if ax is None:
        ax = plt.gca()
    G = nx.DiGraph()
    for ind in ped_ts.individuals():
        time = ped_ts.node(ind.nodes[0]).time
        pop = ped_ts.node(ind.nodes[0]).population
        G.add_node(ind.id, time=time, population=pop)
        for p in ind.parents:
            if p != tskit.NULL:
                G.add_edge(p, ind.id)
    maxtime = int(ped_ts.max_time)
    pos = nx.multipartite_layout(
        G, subset_key="time", align="horizontal", scale=maxtime
    )
    if nx.__version__ == "3.1":
        for coords in pos.values():
            if coords[1] < -2 and coords[1] > -6:
                coords[0] = -coords[0]
    nx.draw_networkx(G, pos, with_labels=False, ax=ax, **kwargs)
    labels = {i.id: "\n".join(str(u) for u in i.nodes) for i in ped_ts.individuals()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=font_size)
    if ts_edge:
        new_G = nx.create_empty_copy(G)
        for e in ped_ts.edges():
            parent = ped_ts.node(e.parent)
            child = ped_ts.node(e.child)
            if parent.individual != tskit.NULL and child.individual != tskit.NULL:
                new_G.add_edge(parent.individual, child.individual)
        nx.draw_networkx_edges(new_G, pos, ax=ax, edge_color=ts_edge, arrowstyle="-")
    if show_axis:
        ax.set_yticks(np.arange(-maxtime, maxtime + 1, 2))
        ax.set_yticklabels(np.arange(maxtime + 1))
        ax.tick_params(labelleft=True)
        ax.set_ylabel("Time ago (generations)")
    if title is not None:
        ax.set_title(title)


def edge_plot(
    ts,
    ax=None,
    title=None,
    use_child_time=None,
    log_time=True,
    linewidths=1,
    plot_hist=False,
    xaxis=True,
    width=None,
    height=None,
    **kwargs,
):
    """
    Plot the edges in a tree sequence. If ax is None, make a new plot, if it is a tuple,
    use the second ax for a histogram
    """

    if width is not None or height is not None:
        default = plt.rcParams["figure.figsize"]
        sz = {"figsize": (width or default[0], height or default[1])}
    else:
        sz = {}
    ax_hist = None
    if ax is None:
        if plot_hist:
            fig, (ax, ax_hist) = plt.subplots(
                1, 2, gridspec_kw={"width_ratios": [5, 1]}, sharey=True, **sz
            )
        else:
            fig, ax = plt.subplots(1, 1, **sz)
    else:
        try:
            ax, ax_hist = ax
        except TypeError:
            pass
    tm = (
        ts.nodes_time[ts.edges_child]
        if use_child_time
        else ts.nodes_time[ts.edges_parent]
    )
    lines = np.array([[ts.edges_left, ts.edges_right], [tm, tm]]).T

    lc = mc.LineCollection(lines, linewidths=linewidths, **kwargs)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0)
    if log_time:
        ax.set_yscale("log")
    ax.set_ylabel(
        f"Time of edge {'child' if use_child_time else 'parent'} ({ts.time_units})"
    )
    if xaxis:
        ax.set_xlabel("Genome position")
    else:
        ax.set_xticks([])
    ax.set_ylim(0.9, None)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if title is not None:
        ax.set_title(title)
    if ax_hist:
        ax_hist.hist(
            tm,
            orientation="horizontal",
            bins=np.logspace(0, np.log10(ts.max_time), 25),
            weights=ts.edges_right - ts.edges_left,
        )
        ax_hist.set_xticks([])


def partial_simplify(
    ts,
    samples=None,
    *args,
    remove_non_coalescent_nodes=None,
    keep_unary_in_individuals=None,
    map_nodes=None,
    filter_individuals=None,
    **kwargs,
):
    """
    Run `simplify` on a tree sequence but keep any nodes that have
    more than one parent or more than one child
    """
    if keep_unary_in_individuals is not None:
        raise ValueError("Cannot use this option with `simplify_remove_pass_through`")
    if filter_individuals is not None:
        raise ValueError("Cannot use this option with `simplify_remove_pass_through`")
    if samples is None:
        samples = ts.samples()
    tables = ts.dump_tables()
    # hack this using `keep_unary in individuals`, by adding each node we want to keep
    # to a new individual
    fake_individual = tables.individuals.add_row()
    keep_nodes = np.zeros(ts.num_nodes, dtype=bool)
    if remove_non_coalescent_nodes:
        _, node_map = ts.simplify(samples, map_nodes=True)  # retains coalescent nodes
        keep_nodes[np.where(node_map != tskit.NULL)] = True
    else:
        keep_nodes[samples] = True
        unique_edges = np.unique(
            np.array([tables.edges.parent, tables.edges.child]).T, axis=0
        )
        for i in (0, 1):  # parent, child
            uniq, count = np.unique(unique_edges[:, i], return_counts=True)
            keep_nodes[uniq[count > 1]] = True
    # Add a new individual for each node we want to keep, if it doesn't have one
    nodes_individual = tables.nodes.individual
    nodes_individual[np.logical_not(keep_nodes)] = tskit.NULL
    nodes_individual[np.logical_and(keep_nodes, nodes_individual == tskit.NULL)] = (
        fake_individual
    )
    tables.nodes.individual = nodes_individual
    node_map = tables.simplify(
        samples,
        *args,
        keep_unary_in_individuals=True,
        filter_individuals=False,
        **kwargs,
    )
    # Remove the fake individual
    tables.individuals.truncate(ts.num_individuals)
    nodes_individual = tables.nodes.individual
    nodes_individual[nodes_individual == fake_individual] = tskit.NULL
    tables.nodes.individual = nodes_individual
    if map_nodes:
        return tables.tree_sequence(), node_map
    else:
        return tables.tree_sequence()


def mutation_labels(ts):
    """
    Return a mapping of mutation ID to "AncestralPosDerived" label
    """
    mut_labels = (
        {}
    )  # An array of labels for the mutations, listing position & allele change
    label = "{}{:g}{}"
    for mut in ts.mutations():  # This entire loop is just to make pretty labels
        site = ts.site(mut.site)
        older_mut = mut.parent >= 0  # is there an older mutation at the same position?
        prev = (
            ts.mutation(mut.parent).derived_state if older_mut else site.ancestral_state
        )
        mut_labels[mut.id] = label.format(prev, site.position, mut.derived_state)
    return mut_labels


# Straight from tskit tests.test_stats
def parse_time_windows(ts, time_windows):
    if time_windows is None:
        time_windows = [0.0, ts.max_root_time]
    return np.array(time_windows)


def windowed_genealogical_nearest_neighbours(
    ts,
    focal,
    reference_sets,
    windows=None,
    time_windows=None,
    span_normalise=True,
    time_normalise=True,
):
    """
    genealogical_nearest_neighbours with support for span- and time-based windows
    """
    reference_set_map = np.full(ts.num_nodes, tskit.NULL, dtype=int)
    for k, reference_set in enumerate(reference_sets):
        for u in reference_set:
            if reference_set_map[u] != tskit.NULL:
                raise ValueError("Duplicate value in reference sets")
            reference_set_map[u] = k

    windows_used = windows is not None
    time_windows_used = time_windows is not None
    windows = ts.parse_windows(windows)
    num_windows = windows.shape[0] - 1
    time_windows = parse_time_windows(ts, time_windows)
    num_time_windows = time_windows.shape[0] - 1
    A = np.zeros((num_windows, num_time_windows, len(focal), len(reference_sets)))
    K = len(reference_sets)
    parent = np.full(ts.num_nodes, tskit.NULL, dtype=int)
    sample_count = np.zeros((ts.num_nodes, K), dtype=int)
    time = ts.tables.nodes.time
    norm = np.zeros((num_windows, num_time_windows, len(focal)))

    # Set the initial conditions.
    for j in range(K):
        sample_count[reference_sets[j], j] = 1

    window_index = 0
    for (t_left, t_right), edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            parent[edge.child] = tskit.NULL
            v = edge.parent
            while v != tskit.NULL:
                sample_count[v] -= sample_count[edge.child]
                v = parent[v]
        for edge in edges_in:
            parent[edge.child] = edge.parent
            v = edge.parent
            while v != tskit.NULL:
                sample_count[v] += sample_count[edge.child]
                v = parent[v]

        # Update the windows
        assert window_index < num_windows
        while windows[window_index] < t_right and window_index + 1 <= num_windows:
            w_left = windows[window_index]
            w_right = windows[window_index + 1]
            left = max(t_left, w_left)
            right = min(t_right, w_right)
            span = right - left
            # Process this tree.
            for j, u in enumerate(focal):
                focal_reference_set = reference_set_map[u]
                delta = int(focal_reference_set != tskit.NULL)
                p = u
                while p != tskit.NULL:
                    total = np.sum(sample_count[p])
                    if total > delta:
                        break
                    p = parent[p]
                if p != tskit.NULL:
                    scale = span / (total - delta)
                    time_index = np.searchsorted(time_windows, time[p]) - 1
                    if 0 <= time_index < num_time_windows:
                        for k in range(len(reference_sets)):
                            n = sample_count[p, k] - int(focal_reference_set == k)
                            A[window_index, time_index, j, k] += n * scale
                        norm[window_index, time_index, j] += span
            assert span > 0
            if w_right <= t_right:
                window_index += 1
            else:
                # This interval crosses a tree boundary, so we update it again
                # in the next tree
                break

    # Reshape norm depending on normalization selected
    # Return NaN when normalisation value is 0
    if span_normalise and time_normalise:
        reshaped_norm = norm.reshape((num_windows, num_time_windows, len(focal), 1))
    elif span_normalise and not time_normalise:
        norm = np.sum(norm, axis=1)
        reshaped_norm = norm.reshape((num_windows, 1, len(focal), 1))
    elif time_normalise and not span_normalise:
        norm = np.sum(norm, axis=0)
        reshaped_norm = norm.reshape((1, num_time_windows, len(focal), 1))

    with np.errstate(invalid="ignore", divide="ignore"):
        A /= reshaped_norm
    A[np.all(A == 0, axis=3)] = np.nan

    # Remove dimension for windows and/or time_windows if parameter is None
    if not windows_used and time_windows_used:
        A = A.reshape((num_time_windows, len(focal), len(reference_sets)))
    elif not time_windows_used and windows_used:
        A = A.reshape((num_windows, len(focal), len(reference_sets)))
    elif not windows_used and not time_windows_used:
        A = A.reshape((len(focal), len(reference_sets)))
    return A


def haplotype_gnn(
    ts,
    focal_ind,
    windows,
    group_sample_sets,
):
    """Calculate the haplotype genealogical nearest neighbours (GNN)
    for the haplotypes of a focal individual"""
    ind = ts.individual(focal_ind)
    A = windowed_genealogical_nearest_neighbours(
        ts, ind.nodes, group_sample_sets, windows=windows
    )
    dflist = []
    for i in range(A.shape[1]):
        x = pd.DataFrame(A[:, i, :])
        x["haplotype"] = i
        x["start"] = windows[0:-1]
        x["end"] = windows[1:]
        dflist.append(x)
    df = pd.concat(dflist)
    df.set_index(["haplotype", "start", "end"], inplace=True)
    return df


def ts2vcz(ts, zarr_file_name):
    # Add imports in here as we don't need them for jupyterlite
    import pysam
    from Bio import bgzf
    if np.any(ts.nodes_individual[ts.samples()] == tskit.NULL):
        raise ValueError("All samples must have an individual")
    samples = set(ts.samples())
    with tempfile.TemporaryDirectory() as tmpdirname:
        vcf_name = os.path.join(tmpdirname, zarr_file_name.replace("/", "") + ".vcf.gz")
        with bgzf.open(vcf_name, "wt") as f:
            ts.write_vcf(f)
        pysam.tabix_index(vcf_name, preset="vcf")
        try:
            subprocess.run([sys.executable, "-m", "bio2zarr", "vcf2zarr", "convert", "--force", vcf_name, zarr_file_name])
        except FileNotFoundError:
            print("Please install bio2zarr to convert VCF to Zarr")
        # set sequence length to match
        z = zarr.open(zarr_file_name)
        z.attrs["sequence_length"] = ts.sequence_length
        if ts.num_populations > 0:
            # add populations
            schema = json.dumps(ts.table_metadata_schemas.population.schema).encode()
            z["populations_metadata_schema"] = schema
            metadata = []
            for pop in ts.populations():
                metadata.append(json.dumps(pop.metadata).encode())
            z["populations_metadata"] = metadata
        individual_population_array = []
        individual_metadata_array = []
        for ind in ts.individuals():
            population = {ts.nodes_population[n] for n in ind.nodes if n in samples}
            if len(population) > 1:
                raise ValueError("Nodes in individuals must belong to a single population")
            if len(population) == 1:
                individual_population_array.append(population.pop())
            if ind.metadata:
                individual_metadata_array.append(json.dumps(ind.metadata).encode())
            else:
                individual_metadata_array.append(b"")
        if len(individual_population_array) > 0:
            schema = json.dumps(ts.table_metadata_schemas.individual.schema).encode()
            z["individuals_metadata_schema"] = schema
            z["individuals_population"] = individual_population_array
            z["individuals_metadata"] = individual_metadata_array
