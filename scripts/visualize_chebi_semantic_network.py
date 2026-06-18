from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import networkx as nx


RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"
OWL_NS = "http://www.w3.org/2002/07/owl#"


def _compact(node: str) -> str:
    text = node
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    if "/" in text:
        return text.rsplit("/", 1)[-1]
    return text


def _display_label(labels: Dict[str, str], node: str, max_len: int = 28) -> str:
    if node in labels:
        label = labels[node]
        return label if len(label) <= max_len else label[: max_len - 1] + "~"
    compact = _compact(node)
    return compact if len(compact) <= max_len else compact[: max_len - 1] + "~"


def _parse_chebi_graph(owl_path: Path) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    labels: Dict[str, str] = {}
    children_of: Dict[str, List[str]] = defaultdict(list)

    rdf_about = "{" + RDF_NS + "}about"
    rdf_resource = "{" + RDF_NS + "}resource"
    tag_label = "{" + RDFS_NS + "}label"
    tag_subclass = "{" + RDFS_NS + "}subClassOf"
    class_tags = {"{" + OWL_NS + "}Class", "{" + RDF_NS + "}Description"}

    print("[INFO] Streaming RDF/XML parse started")
    for _, elem in ET.iterparse(str(owl_path), events=("end",)):
        if elem.tag not in class_tags:
            continue

        subject = elem.attrib.get(rdf_about)
        if not subject:
            elem.clear()
            continue

        for child in elem:
            if child.tag == tag_label and child.text:
                labels[subject] = child.text.strip()
            elif child.tag == tag_subclass:
                parent = child.attrib.get(rdf_resource)
                if parent:
                    children_of[parent].append(subject)

        elem.clear()

    return labels, children_of


def _find_root_by_label(labels: Dict[str, str], root_label: str) -> str | None:
    norm = root_label.strip().lower()
    for iri, label in labels.items():
        if label.strip().lower() == norm:
            return iri
    return None


def _bfs_neighborhood(
    children_of: Dict[str, List[str]],
    root: str,
    max_depth: int,
    max_nodes: int,
) -> Tuple[Set[str], List[Tuple[str, str]], Dict[str, int]]:
    visited: Set[str] = {root}
    level: Dict[str, int] = {root: 0}
    q: deque[str] = deque([root])

    while q and len(visited) < max_nodes:
        node = q.popleft()
        depth = level[node]
        if depth >= max_depth:
            continue

        for child in children_of.get(node, []):
            if child in visited:
                continue
            visited.add(child)
            level[child] = depth + 1
            q.append(child)
            if len(visited) >= max_nodes:
                break

    kept_edges: List[Tuple[str, str]] = []
    for parent, children in children_of.items():
        if parent not in visited:
            continue
        for child in children:
            if child in visited:
                kept_edges.append((parent, child))

    return visited, kept_edges, level


def _layout_by_level(level: Dict[str, int]) -> Dict[str, Tuple[float, float]]:
    by_level: Dict[int, List[str]] = defaultdict(list)
    for node, depth in level.items():
        by_level[depth].append(node)

    pos: Dict[str, Tuple[float, float]] = {}
    for depth in sorted(by_level):
        nodes = sorted(by_level[depth], key=lambda n: str(n))
        count = len(nodes)
        for idx, node in enumerate(nodes):
            x = 0.5 if count == 1 else idx / (count - 1)
            y = -float(depth)
            pos[node] = (x, y)
    return pos


def _layout_by_spring(nodes: Sequence[str], edges: Sequence[Tuple[str, str]]) -> Dict[str, Tuple[float, float]]:
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    if g.number_of_nodes() == 1:
        only = next(iter(g.nodes()))
        return {only: (0.5, 0.0)}

    # Larger k spreads nodes further apart, reducing overlap in dense graph views.
    n = g.number_of_nodes()
    k = 1.8 / max(1.0, n ** 0.5)
    return nx.spring_layout(g, seed=42, k=k, iterations=220)


def _resolve_node_collisions(
    pos: Dict[str, Tuple[float, float]],
    level: Dict[str, int],
    iterations: int,
    min_dist: float,
) -> Dict[str, Tuple[float, float]]:
    nodes = list(pos.keys())
    xy = {n: [pos[n][0], pos[n][1]] for n in nodes}
    eps = 1e-9

    for _ in range(max(0, iterations)):
        moved = False
        for i in range(len(nodes)):
            ni = nodes[i]
            xi, yi = xy[ni]
            for j in range(i + 1, len(nodes)):
                nj = nodes[j]
                xj, yj = xy[nj]
                dx = xi - xj
                dy = yi - yj
                dist2 = dx * dx + dy * dy
                if dist2 <= eps:
                    dx, dy = 1e-4, 0.0
                    dist2 = dx * dx + dy * dy

                dist = dist2 ** 0.5
                if dist >= min_dist:
                    continue

                # Push both nodes away from each other.
                push = 0.5 * (min_dist - dist)
                ux = dx / dist
                uy = dy / dist

                # Keep shallow semantic nodes slightly steadier than deeper nodes.
                wi = 1.0 / max(1, level.get(ni, 1) + 1)
                wj = 1.0 / max(1, level.get(nj, 1) + 1)
                wsum = wi + wj
                wi, wj = wi / wsum, wj / wsum

                xy[ni][0] += ux * push * wj
                xy[ni][1] += uy * push * wj
                xy[nj][0] -= ux * push * wi
                xy[nj][1] -= uy * push * wi
                moved = True

        if not moved:
            break

    return {n: (xy[n][0], xy[n][1]) for n in nodes}


def _build_undirected_neighbor_map(edges: Sequence[Tuple[str, str]]) -> Dict[str, Set[str]]:
    nbrs: Dict[str, Set[str]] = defaultdict(set)
    for a, b in edges:
        nbrs[a].add(b)
        nbrs[b].add(a)
    return nbrs


def _children_of_parent(edges: Sequence[Tuple[str, str]]) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = defaultdict(list)
    for parent, child in edges:
        children[parent].append(child)
    return children


def _augment_dense_edges(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    sibling_cap: int,
) -> List[Tuple[str, str]]:
    node_set = set(nodes)
    edge_set: Set[Tuple[str, str]] = set(edges)
    children_map = _children_of_parent(edges)

    # Add sibling-to-sibling links under the same parent for denser visualization.
    for parent, children in children_map.items():
        kept_children = [c for c in sorted(children, key=str) if c in node_set]
        if len(kept_children) < 2:
            continue

        if sibling_cap > 0:
            kept_children = kept_children[:sibling_cap]

        for i in range(len(kept_children)):
            for j in range(i + 1, len(kept_children)):
                a = kept_children[i]
                b = kept_children[j]
                edge_set.add((a, b))

    return list(edge_set)


def _compress_representative_nodes(
    root: str,
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    level: Dict[str, int],
    target_nodes: int,
) -> Tuple[List[str], List[Tuple[str, str]], Dict[str, int]]:
    if len(nodes) <= target_nodes:
        return list(nodes), list(edges), dict(level)

    by_level: Dict[int, List[str]] = defaultdict(list)
    for node in nodes:
        by_level[level[node]].append(node)

    max_depth = max(by_level.keys())
    keep: Set[str] = {root}

    root_children = sorted([child for parent, child in edges if parent == root], key=str)
    keep.update(root_children)

    # Root and all direct children are mandatory for explanatory semantic figures.
    min_required = 1 + len(root_children)
    target_nodes = max(target_nodes, min_required)

    # Keep a small, representative set from each level to preserve semantic hierarchy.
    for depth in range(1, max_depth + 1):
        level_nodes = sorted(by_level.get(depth, []), key=lambda n: str(n))
        if not level_nodes:
            continue

        budget = max(2, target_nodes // max(1, max_depth + 1))
        step = max(1, len(level_nodes) // budget)
        selected = level_nodes[::step][:budget]
        keep.update(selected)

    if len(keep) > target_nodes:
        # Trim deepest nodes first if we overshoot the target.
        sorted_keep = sorted(keep, key=lambda n: (level[n], str(n)))
        must_keep = [n for n in sorted_keep if n == root or level[n] == 1]
        optional = [n for n in sorted_keep if n not in must_keep]
        keep = set((must_keep + optional)[:target_nodes])

    nbrs = _build_undirected_neighbor_map(edges)
    if root not in keep:
        keep.add(root)

    # Ensure connectivity by adding shortest connector paths from root.
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    for node in list(keep):
        if node == root:
            continue
        try:
            path = nx.shortest_path(g, source=root, target=node)
            keep.update(path)
        except nx.NetworkXNoPath:
            continue

    if len(keep) > target_nodes:
        # Final trim while preserving root and shallow nodes.
        sorted_keep = sorted(keep, key=lambda n: (level[n], str(n)))
        fixed = [n for n in sorted_keep if n == root or level[n] <= 1]
        rest = [n for n in sorted_keep if n not in fixed]
        keep = set((fixed + rest)[:target_nodes])

    kept_edges = [(a, b) for a, b in edges if a in keep and b in keep]

    # Keep only the connected component containing root.
    sub = nx.Graph()
    sub.add_nodes_from(keep)
    sub.add_edges_from(kept_edges)
    if root in sub:
        component = nx.node_connected_component(sub, root)
        keep = set(component)
        kept_edges = [(a, b) for a, b in kept_edges if a in keep and b in keep]

    new_nodes = sorted(keep, key=lambda n: (level[n], str(n)))
    new_level = {n: level[n] for n in new_nodes}
    return new_nodes, kept_edges, new_level


def _select_labeled_nodes(
    nodes: Sequence[str],
    level: Dict[str, int],
    label_mode: str,
    label_max_depth: int,
    label_every: int,
) -> Set[str]:
    if label_mode == "all":
        return set(nodes)

    # Sparse mode: keep all labels on shallow levels, subsample deeper levels.
    by_level: Dict[int, List[str]] = defaultdict(list)
    for node in nodes:
        by_level[level[node]].append(node)

    keep: Set[str] = set()
    for depth, depth_nodes in by_level.items():
        depth_nodes_sorted = sorted(depth_nodes, key=lambda n: str(n))
        if depth <= label_max_depth:
            keep.update(depth_nodes_sorted)
            continue
        step = max(1, label_every)
        keep.update(depth_nodes_sorted[::step])

    return keep


def _draw_figure(
    labels: Dict[str, str],
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str]],
    level: Dict[str, int],
    out_png: Path,
    out_svg: Path,
    title: str,
    layout_mode: str,
    label_mode: str,
    label_max_depth: int,
    label_every: int,
    label_font_size: int,
    canvas_pad: float,
    collision_iters: int,
    collision_min_dist: float,
) -> None:
    if layout_mode == "spring":
        pos = _layout_by_spring(nodes, edges)
    else:
        pos = _layout_by_level(level)

    pos = _resolve_node_collisions(
        pos=pos,
        level=level,
        iterations=collision_iters,
        min_dist=collision_min_dist,
    )
    labeled_nodes = _select_labeled_nodes(
        nodes=nodes,
        level=level,
        label_mode=label_mode,
        label_max_depth=label_max_depth,
        label_every=label_every,
    )

    fig, ax = plt.subplots(figsize=(13, 8), dpi=220)
    ax.set_title(title, fontsize=14, pad=12)
    ax.axis("off")

    for parent, child in edges:
        x1, y1 = pos[parent]
        x2, y2 = pos[child]
        ax.plot([x1, x2], [y1, y2], color="#6C757D", alpha=0.45, linewidth=0.8, zorder=1)

    node_count = max(1, len(nodes))
    size_scale = min(1.0, 40.0 / node_count + 0.45)

    for node in nodes:
        x, y = pos[node]
        depth = level[node]
        base = max(22, 130 - depth * 16)
        size = max(18, base * size_scale)
        ax.scatter([x], [y], s=size, color="#145A32", alpha=0.95, zorder=2)
        if node in labeled_nodes:
            ax.text(
                x,
                y + 0.07,
                _display_label(labels, node),
                fontsize=label_font_size,
                ha="center",
                va="bottom",
                color="#111111",
                zorder=3,
            )

    # Tighten view bounds so the graph occupies the canvas with minimal whitespace.
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    pad_x = span_x * max(0.0, canvas_pad)
    pad_y = span_y * max(0.0, canvas_pad)

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.margins(0.0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize CHebi semantic network as a paper-ready figure")
    parser.add_argument("--owl", default="ontology/chebi.owl", help="Path to ChEBI OWL file")
    parser.add_argument("--root-label", default="chemical entity", help="Root concept label")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum BFS depth")
    parser.add_argument("--max-nodes", type=int, default=120, help="Maximum node count")
    parser.add_argument(
        "--target-nodes",
        type=int,
        default=0,
        help="If >0, compress to an example-size connected subgraph with approximately this many nodes",
    )
    parser.add_argument(
        "--layout-mode",
        default="spring",
        choices=["spring", "hierarchy"],
        help="Graph layout mode. spring gives a graph-like visual, hierarchy keeps tree-style",
    )
    parser.add_argument(
        "--dense-links",
        action="store_true",
        help="Add sibling links to make the graph visually denser",
    )
    parser.add_argument(
        "--dense-sibling-cap",
        type=int,
        default=16,
        help="Maximum number of siblings per parent to use for dense-link augmentation",
    )
    parser.add_argument("--out-dir", default="output/figures", help="Output directory")
    parser.add_argument(
        "--label-mode",
        default="all",
        choices=["all", "sparse"],
        help="Labeling mode: all nodes or sparse labeling for paper readability",
    )
    parser.add_argument(
        "--label-max-depth",
        type=int,
        default=2,
        help="In sparse mode, fully label nodes up to this depth",
    )
    parser.add_argument(
        "--label-every",
        type=int,
        default=4,
        help="In sparse mode, label every Nth node for depths > label-max-depth",
    )
    parser.add_argument("--label-font-size", type=int, default=7, help="Label font size")
    parser.add_argument(
        "--canvas-pad",
        type=float,
        default=0.03,
        help="Relative inner padding around plotted nodes (lower means tighter fill)",
    )
    parser.add_argument(
        "--collision-iters",
        type=int,
        default=80,
        help="Post-layout collision relaxation iterations for reducing node overlap",
    )
    parser.add_argument(
        "--collision-min-dist",
        type=float,
        default=0.075,
        help="Minimum center distance to enforce between nodes in layout space",
    )
    args = parser.parse_args()

    owl_path = Path(args.owl)
    if not owl_path.exists():
        raise FileNotFoundError(f"OWL file not found: {owl_path}")

    print(f"[INFO] Loading ontology: {owl_path}")
    labels, children_of = _parse_chebi_graph(owl_path)

    root = _find_root_by_label(labels, args.root_label)
    if root is None:
        raise ValueError(f"Could not find root concept with label: {args.root_label}")

    print(f"[INFO] Root concept: {_display_label(labels, root, max_len=80)} ({root})")

    nodes_set, edges, level = _bfs_neighborhood(
        children_of=children_of,
        root=root,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
    )

    nodes = sorted(nodes_set, key=lambda n: (level[n], str(n)))

    if args.target_nodes > 0:
        nodes, edges, level = _compress_representative_nodes(
            root=root,
            nodes=nodes,
            edges=edges,
            level=level,
            target_nodes=args.target_nodes,
        )

    if args.dense_links:
        edges = _augment_dense_edges(
            nodes=nodes,
            edges=edges,
            sibling_cap=args.dense_sibling_cap,
        )

    out_dir = Path(args.out_dir)
    out_png = out_dir / "chebi_semantic_network.png"
    out_svg = out_dir / "chebi_semantic_network.svg"

    title = f"ChEBI Semantic Network (root: {_display_label(labels, root)}, depth<={args.max_depth})"
    _draw_figure(
        labels,
        nodes,
        edges,
        level,
        out_png,
        out_svg,
        title,
        layout_mode=args.layout_mode,
        label_mode=args.label_mode,
        label_max_depth=args.label_max_depth,
        label_every=args.label_every,
        label_font_size=args.label_font_size,
        canvas_pad=args.canvas_pad,
        collision_iters=args.collision_iters,
        collision_min_dist=args.collision_min_dist,
    )

    print(f"[DONE] Nodes: {len(nodes)}")
    print(f"[DONE] Edges: {len(edges)}")
    print(f"[DONE] Layout mode: {args.layout_mode}")
    print(f"[DONE] Dense links: {args.dense_links}")
    print(f"[DONE] Label mode: {args.label_mode}")
    print(f"[DONE] PNG:  {out_png}")
    print(f"[DONE] SVG:  {out_svg}")


if __name__ == "__main__":
    main()