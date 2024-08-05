import collections
import itertools
import os
import sys
from datetime import datetime

from matplotlib import pyplot as plt
from matplotlib import collections  as mc

import networkx as nx
import numpy as np
import tskit
import tqdm
import yaml
from IPython.core.display import HTML
from jupyterquiz import display_quiz

from model.coal import add_mutations
from model.coal import make_tree
from model.draw import draw_tree

path = os.path.dirname(os.path.normpath(__file__))


def load_quiz(section):
    with open(os.path.join(path, "quiz.yaml")) as fh:
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
    styles = open(os.path.join(path, "styles/custom.css")).read()
    css = f"<style>{styles}</style>"
    js = "<script src='https://d3js.org/d3.v7.min.js'></script>"
    # See https://github.com/jupyterlite/jupyterlite/issues/407#issuecomment-1353088447
    html_text = [
        """<table style="width: 100%;"><tr>
        <td style="text-align: left;">✅ Your notebook is ready to go!</td>""",  # => 0
        """<td style="text-align: right;">
        <button type="button" id="button_for_indexeddb">Clear JupyterLite local storage
        </button></td>""",   # => 1 (omit if not in jlite)
        "</tr></table>",   # => 2
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
            return HTML(self.js + self.css + "".join([self.html_text[0], self.html_text[2]]))


class HOWTO(Workbook):
    def __init__(self):
        super().__init__()
        self.quiz["day"][0]["answers"][0]["value"] = datetime.today().day


class WrightFisher(Workbook):
    def __init__(self):
        super().__init__()


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


class CoalescentHandson(Workbook):
    def __init__(self):
        super().__init__()
        ancestors = [5, 6, 4, 4, 5, 6]
        branches = [0.3, 0.9, 0.1, 0.1, 0.2, 0.6, 0.9]
        self.tree = make_tree(ancestors, branches)
        mutations = [0, 2, 0, 1, 1, 2, 0]
        add_mutations(self.tree, mutations)

    def _draw_coalescent(self):
        d = draw_tree(
            self.tree,
            node_labels=True,
            show_internal=True,
            node_size=3,
            height=400,
            mutation_size=0,
            jitter_label=(0, 20),
        )
        return d

    def _draw_coalescent_w_mutations(self):
        d = draw_tree(
            self.tree,
            node_labels=True,
            show_internal=True,
            node_size=3,
            height=400,
            mutation_size=5,
            jitter_label=(0, 20),
        )
        return d

    def draw(self, which):
        if which == "coalescent_tree":
            return self._draw_coalescent()
        elif which == "coalescent_tree_w_mutations":
            return self._draw_coalescent_w_mutations()


class MsprimeSimulations(Workbook):
    def __init__(self):
        super().__init__()


def setup_msprime_simulations():
    return MsprimeSimulations()


def setup_coalescent_handson():
    return CoalescentHandson()


def setup_wright_fisher():
    return WrightFisher()

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
            self.current_population = self.reproduce(self.current_population, neg_gens-1)

        # On output, rebase the times so the current generation is at time 0
        self.tables.nodes.time += gens
        # Reorder the nodes in time order, youngest first
        time_order = np.argsort(self.tables.nodes.time, kind='stable')
        node_map = {u: i for i, u in enumerate(time_order)}
        self.tables.subset(time_order)
        for k in self.current_population.keys():
            self.current_population[k] = [node_map[v] for v in self.current_population[k]]
        self.tables.sort()  # Sort edges into canonical order, required for converting to a tree seq

        if simplify:
            if "samples" not in kwargs:
                kwargs["samples"] = [u for nodes in self.current_population.values() for u in nodes]
            node_map = self.tables.simplify(**kwargs)
            for k in self.current_population.keys():
                self.current_population[k] = [node_map[v] for v in self.current_population[k]]
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
        breakpoint_positions = np.unique([0, *self.rec_rng.integers(L, size=num_breakpoints), L])
        choose_genome = 0
        for left, right in zip(breakpoint_positions[:-1], breakpoint_positions[1:]):
            self.tables.edges.add_row(
                left=left,
                right=right,
                parent=randomly_ordered_parent_nodes[choose_genome],
                child=child_node,
            )
            choose_genome = 1 if choose_genome == 0 else 0

    def __init__(self, population_size, seq_len=1000, recombination_rate=1e-8, random_seed=427):
        self.recombination_rate = recombination_rate
        self.rec_rng = np.random.default_rng(seed=random_seed)
        super().__init__(population_size, seq_len, random_seed)
## Functions for drawing & plotting

def basic_genealogy_viz(tables_or_ts, ax=None, *, show_node_ids=True, show_individuals=None, title=None):
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
    ax=plt.gca()
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
    node_dict = {k: node_dict[k] for k in sorted(node_dict.keys(), key=lambda x: node_dict[x])}
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
      ax.annotate(i, (node_pos_x[i], y-tweak_y), ha="center", va="top")
  if len(individual_pos) and show_individuals:
      individual_pos = np.array([np.mean(v, axis=0) for v in individual_pos.values()])
      ax.scatter(individual_pos[:,0], individual_pos[:,1]-tweak_y, facecolors='none', edgecolors="0.9", marker="H", s=1400)
  for edge in tables.edges:
    x = node_pos_x[edge.child]
    dx = node_pos_x[edge.parent] - x
    ax.arrow(x, tables.nodes[edge.child].time, dx, 1, color="lightblue")
  ax.set_xticks([])
  max_x = tables.nodes.time.max() + tweak_y * 3
  ax.set_ylim(None, max_x)
  ax.set_yticks(np.arange(tables.nodes.time.max()+1))
  ax.set_ylabel("Time (generations ago)")
  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  if title is not None:
    ax.set_title(title)


rotated_label_style = (
    ".node > .lab {font-size: 80%}"
    ".leaf > .lab {text-anchor: start; transform: rotate(90deg) translate(6px)}"
)

def draw_pedigree(ped_ts, ax=None, title=None, show_axis=False, font_size=None, ts_edge=None, **kwargs):
    # If ts_edge is None, do not overlay tskit edges, otherwise specify a color
    # or "black" if True
    if ts_edge is True:
       ts_edge = "black"
    if ax is None:
        ax=plt.gca()
    G = nx.DiGraph()
    for ind in ped_ts.individuals():
        time = ped_ts.node(ind.nodes[0]).time
        pop = ped_ts.node(ind.nodes[0]).population
        G.add_node(ind.id, time=time, population=pop)
        for p in ind.parents:
            if p != tskit.NULL:
                G.add_edge(p, ind.id)
    maxtime = int(ped_ts.max_time)
    pos = nx.multipartite_layout(G, subset_key="time", align="horizontal", scale=maxtime)
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
        ax.set_ylabel("Time (generations ago)")
    if title is not None:
        ax.set_title(title)


def edge_plot(ts, ax=None, title=None, use_child_time=None, log_time=True, linewidths=1, plot_hist=False, xaxis=True, **kwargs):
    """
    Plot the edges in a tree sequence, 
    """
    if ax is None:
        if plot_hist:
            fig, (ax, ax_hist) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, sharey=True) 
        else:
            ax=plt.gca()
            ax_hist=None
        
    tm = ts.nodes_time[ts.edges_child] if use_child_time else ts.nodes_time[ts.edges_parent]
    lines = np.array([[ts.edges_left, ts.edges_right], [tm, tm]]).T

    lc = mc.LineCollection(lines, linewidths=linewidths, **kwargs)
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0)
    if log_time:
        ax.set_yscale("log")
    ax.set_ylabel(f"Time of edge {'child' if use_child_time else 'parent'} ({ts.time_units})")
    if xaxis:
        ax.set_xlabel("Genome position")
    else:
        ax.set_xticks([])
    ax.set_ylim(0.9, None)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
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
    ts, samples=None, *args, remove_all_locally_unary=None,
    keep_unary_in_individuals=None, map_nodes=None, filter_individuals=None, **kwargs):
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
    if remove_all_locally_unary:
        _, node_map = ts.simplify(samples, map_nodes=True)  # retains coalescent nodes
        keep_nodes[np.where(node_map != tskit.NULL)] = True
    else:
        keep_nodes[samples] = True
        unique_edges = np.unique(np.array([tables.edges.parent, tables.edges.child]).T, axis=0)
        for i in (0, 1):  # parent, child
            uniq, count = np.unique(unique_edges[:, i], return_counts=True)
            keep_nodes[uniq[count > 1]] = True
    # Add a new individual for each node we want to keep, if it doesn't have one
    nodes_individual = tables.nodes.individual
    nodes_individual[np.logical_not(keep_nodes)] = tskit.NULL
    nodes_individual[np.logical_and(keep_nodes, nodes_individual==tskit.NULL)] = fake_individual
    tables.nodes.individual=nodes_individual
    node_map = tables.simplify(
        samples,
        *args, 
        keep_unary_in_individuals=True,
        filter_individuals=False,
        **kwargs)
    # Remove the fake individual
    tables.individuals.truncate(ts.num_individuals)
    nodes_individual = tables.nodes.individual
    nodes_individual[nodes_individual == fake_individual] = tskit.NULL
    tables.nodes.individual=nodes_individual
    if map_nodes:
        return tables.tree_sequence(), node_map
    else:
        return tables.tree_sequence()
    


def mutation_labels(ts):
    """
    Return a mapping of mutation ID to "AncestralPosDerived" label
    """
    mut_labels = {}  # An array of labels for the mutations, listing position & allele change
    l = "{}{:g}{}"
    for mut in ts.mutations():  # This entire loop is just to make pretty labels
        site = ts.site(mut.site)
        older_mut = mut.parent >= 0  # is there an older mutation at the same position?
        prev = ts.mutation(mut.parent).derived_state if older_mut else site.ancestral_state
        mut_labels[mut.id] = l.format(prev, site.position, mut.derived_state)
    return mut_labels


