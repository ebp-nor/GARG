import drawsvg as dw
import numpy as np
from model.coal import Coordinate
from model.coal import Tree
from model.coal import add_mutations
from model.coal import make_tree


def draw_tree(
    tree: Tree,
    *,
    width=400,
    height=200,
    id_prefix=None,
    node_labels=False,
    show_internal=False,
    font_size=18,
    node_size=0,
    mutation_size=3,
    jitter_label=(0, 0),
    **kwargs
):
    # Traverse tree to get x coordinates
    x = np.array([n.coords.x for n in tree.nodes])
    y = np.array([n.coords.y for n in tree.nodes])
    # Scale x by 0.9 * width
    x = x - min(x)
    scalex = 0.9 * width / max(x)
    x = x * scalex + 0.05 * width
    # Scale y by 0.9 * height and reverse orientation
    y = abs(y - max(y))
    scaley = 0.8 * height / max(y)
    y = y * scaley + 0.1 * height

    # Update the plotting coordinates in new system
    for i, _ in enumerate(tree.nodes):
        tree.nodes[i].plot_coords = Coordinate(x=x[i], y=y[i])
    d = dw.Drawing(width, height, id_prefix=id_prefix)
    for n in tree.nodes:
        d.append(
            dw.Circle(
                n.plot_coords.x,
                n.plot_coords.y,
                node_size,
                fill="black",
                stroke="black",
            )
        )
        ancestor = n.ancestor
        if ancestor is not None:
            d.append(
                dw.Line(
                    n.plot_coords.x,
                    n.plot_coords.y,
                    ancestor.plot_coords.x,
                    ancestor.plot_coords.y,
                    stroke="black",
                )
            )
        # Add labels if needed
        show_labels = n.isleaf or (not n.isleaf and show_internal)
        if node_labels and show_labels:
            # Calculate node placement
            p = n.plot_coords + Coordinate(x=jitter_label[0], y=jitter_label[1])
            d.append(dw.Text(str(n.label), font_size, p.x, p.y, center=True))
        # Add mutations
        for i in range(n.mutations):
            mut = (n.ancestor.plot_coords - n.plot_coords) * (i + 1) / (
                n.mutations + 1
            ) + n.plot_coords
            upper_left = Coordinate(
                mut.x - mutation_size / 2, mut.y - mutation_size / 2
            )
            d.append(
                dw.Rectangle(
                    upper_left.x,
                    upper_left.y,
                    mutation_size,
                    mutation_size,
                    fill="black",
                    stroke="black",
                )
            )
    return d


def plot_ancestry(ancestors, branches, mutations=None, **kwargs):
    tree = make_tree(ancestors, branches)
    if mutations is not None:
        tree = add_mutations(tree, mutations)
    return draw_tree(tree, **kwargs)
