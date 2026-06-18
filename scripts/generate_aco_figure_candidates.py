from __future__ import annotations

import math
import random
import runpy
from collections import defaultdict
from pathlib import Path
import textwrap
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx


Node = str
Edge = Tuple[Node, Node]


def _load_graph_builders() -> dict:
    return runpy.run_path("scripts/visualize_chebi_semantic_network.py")


def _canonical_edge(a: Node, b: Node) -> Edge:
    return (a, b) if a <= b else (b, a)


def _build_semantic_subgraph(
    owl_path: Path,
    root_label: str,
    max_depth: int,
    max_nodes: int,
    target_nodes: int,
) -> Tuple[Dict[str, str], List[Node], List[Edge], Dict[Node, int], Node]:
    mod = _load_graph_builders()
    parse = mod["_parse_chebi_graph"]
    find_root = mod["_find_root_by_label"]
    bfs = mod["_bfs_neighborhood"]
    compress = mod["_compress_representative_nodes"]
    dense = mod["_augment_dense_edges"]

    labels, children_of = parse(owl_path)
    root = find_root(labels, root_label)
    if root is None:
        raise ValueError(f"Root label not found: {root_label}")

    nodes_set, tree_edges, level = bfs(children_of, root, max_depth, max_nodes)
    nodes = sorted(nodes_set, key=lambda n: (level[n], n))
    nodes, tree_edges, level = compress(root, nodes, tree_edges, level, target_nodes)
    edges = dense(nodes, tree_edges, sibling_cap=18)

    # Force undirected edge uniqueness for traversal and plotting stability.
    uniq = {_canonical_edge(a, b) for a, b in edges if a != b}
    edges = sorted(uniq)
    return labels, nodes, edges, level, root


def _eta(level: Dict[Node, int], cur: Node, nxt: Node) -> float:
    # Mild preference for moving down one level; still allows lateral exploration.
    dcur = level.get(cur, 0)
    dnxt = level.get(nxt, 0)
    if dnxt == dcur + 1:
        return 2.0
    if dnxt == dcur:
        return 1.3
    return 1.0


def _sample_next(
    neighbors: Sequence[Node],
    cur: Node,
    level: Dict[Node, int],
    pheromone: Dict[Edge, float],
    alpha: float,
    beta: float,
) -> Node:
    weights = []
    for nxt in neighbors:
        e = _canonical_edge(cur, nxt)
        tau = max(1e-8, pheromone.get(e, 1.0))
        weight = (tau ** alpha) * (_eta(level, cur, nxt) ** beta)
        weights.append(weight)

    total = sum(weights)
    if total <= 0.0:
        return random.choice(list(neighbors))

    r = random.random() * total
    c = 0.0
    for nxt, w in zip(neighbors, weights):
        c += w
        if c >= r:
            return nxt
    return neighbors[-1]


def _dl_token(label: str) -> str:
    token = "".join(ch if ch.isalnum() else "_" for ch in label.strip())
    token = "_".join([p for p in token.split("_") if p])
    return token[:42] if token else "Concept"


def _path_to_dl_rows(path: Sequence[Node], labels: Dict[str, str], level: Dict[Node, int]) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    atoms: List[str] = []
    for i, n in enumerate(path[1:], start=1):
        lbl = labels.get(n, n.split("/")[-1]).strip()
        depth = str(level.get(n, i))
        atom = f"∃hasConcept.{_dl_token(lbl)}"
        atoms.append(atom)
        cumulative = " ⊓ ".join(atoms)
        rows.append((f"t{i}", lbl, atom, cumulative + f"  (d={depth})"))

    if not rows:
        rows.append(("t1", "root-descendant", "∃hasConcept.RootDescendant", "∃hasConcept.RootDescendant  (d=1)"))
    return rows


def _simulate_aco(
    nodes: Sequence[Node],
    edges: Sequence[Edge],
    level: Dict[Node, int],
    root: Node,
    labels: Dict[str, str],
    generations: int = 6,
    ants_per_gen: int = 30,
    max_steps: int = 6,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.2,
) -> dict:
    random.seed(42)

    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    pheromone: Dict[Edge, float] = {e: 1.0 for e in edges}
    edge_visits: Dict[Edge, int] = defaultdict(int)
    snapshots = []
    all_paths = []

    def score(path: Sequence[Node]) -> float:
        if not path:
            return 0.0
        unique = len(set(path))
        depth_gain = sum(level.get(n, 0) for n in set(path)) / max(1, unique)
        return unique + 0.7 * depth_gain

    for gen in range(generations):
        gen_paths = []
        for _ in range(ants_per_gen):
            cur = root
            path = [cur]
            visited = {cur}
            for _ in range(max_steps):
                nbrs = list(g.neighbors(cur))
                if not nbrs:
                    break

                # Soft avoid immediate revisits to improve rule diversity.
                candidates = [n for n in nbrs if n not in visited]
                if not candidates:
                    candidates = nbrs

                nxt = _sample_next(candidates, cur, level, pheromone, alpha, beta)
                e = _canonical_edge(cur, nxt)
                edge_visits[e] += 1
                path.append(nxt)
                visited.add(nxt)
                cur = nxt

            fit = score(path)
            gen_paths.append((path, fit))
            all_paths.append((path, fit))

        # Evaporation
        for e in pheromone:
            pheromone[e] = max(0.05, pheromone[e] * (1.0 - evaporation))

        # Elite deposition
        gen_paths.sort(key=lambda x: x[1], reverse=True)
        elite = gen_paths[: max(1, len(gen_paths) // 4)]
        for path, fit in elite:
            dep = fit / max(1, len(path) - 1)
            for i in range(len(path) - 1):
                e = _canonical_edge(path[i], path[i + 1])
                pheromone[e] = pheromone.get(e, 0.05) + dep

        if gen in (0, 1, generations // 2, generations - 1):
            snapshots.append((gen, dict(pheromone)))

    all_paths.sort(key=lambda x: x[1], reverse=True)
    top_paths = all_paths[:3]
    top_dl_rows = [_path_to_dl_rows(p, labels, level) for p, _ in top_paths]

    return {
        "graph": g,
        "pheromone": pheromone,
        "edge_visits": edge_visits,
        "snapshots": snapshots,
        "top_paths": [p for p, _ in top_paths],
        "top_dl_rows": top_dl_rows,
    }


def _edge_widths(values: Dict[Edge, float], scale: float = 4.0) -> Dict[Edge, float]:
    if not values:
        return {}
    vmin = min(values.values())
    vmax = max(values.values())
    span = max(1e-8, vmax - vmin)
    return {e: 0.6 + scale * ((v - vmin) / span) for e, v in values.items()}


def _fmt_cum(atoms_list: List[str], depth_str: str) -> str:
    """Abbreviate long cumulative expressions to at most 2 display lines."""
    def _short(atom: str) -> str:
        return atom if len(atom) <= 30 else atom[:27] + "..."

    if len(atoms_list) == 1:
        return f"{_short(atoms_list[0])} {depth_str}".strip()
    if len(atoms_list) == 2:
        return f"{_short(atoms_list[0])} ⊓\n{_short(atoms_list[1])} {depth_str}".strip()
    # 3+ atoms: first ⊓ ... ⊓ last  (always 2 lines)
    return f"{_short(atoms_list[0])} ⊓ ...\n⊓ {_short(atoms_list[-1])} {depth_str}".strip()


def _draw_candidate_1(
    output_path: Path,
    g: nx.Graph,
    pos: Dict[Node, Tuple[float, float]],
    top_path: Sequence[Node],
    dl_rows: Sequence[Tuple[str, str, str, str]],
) -> None:
    # Same overall figure size as the reference version.
    # gridspec keeps A/B/C titles at the same vertical level.
    # Table row heights are set per-row based on actual line count so
    # text never overflows cell boundaries.
    fig = plt.figure(figsize=(17.5, 5.6), dpi=220)
    gs = fig.add_gridspec(
        1, 3, width_ratios=[1.1, 1.1, 1.8],
        left=0.02, right=0.99, top=0.93, bottom=0.04,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    nx.draw_networkx_edges(g, pos, ax=ax1, alpha=0.35, edge_color="#888888", width=0.8)
    nx.draw_networkx_nodes(g, pos, ax=ax1, node_size=70, node_color="#2E8B57", alpha=0.9)
    ax1.set_title("A) Semantic Network", pad=4)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    nx.draw_networkx_edges(g, pos, ax=ax2, alpha=0.22, edge_color="#BBBBBB", width=0.7)
    nx.draw_networkx_nodes(g, pos, ax=ax2, node_size=70, node_color="#D9D9D9", alpha=0.9)
    path_edges = [
        _canonical_edge(top_path[i], top_path[i + 1]) for i in range(len(top_path) - 1)
    ]
    nx.draw_networkx_edges(g, pos, ax=ax2, edgelist=path_edges, edge_color="#D62728", width=2.8)
    nx.draw_networkx_nodes(g, pos, ax=ax2, nodelist=list(top_path), node_size=110, node_color="#D62728")
    for i, n in enumerate(top_path, start=1):
        x, y = pos[n]
        ax2.text(x, y, str(i), fontsize=7, color="white", ha="center", va="center")
    ax2.set_title("B) Single Ant Path", pad=4)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    ax3.set_title("C) Description Logic Rule Trace", pad=4)

    col_labels = ["Step", "Concept", "DL Atom", "Cumulative Expression"]
    wrapped_rows = []
    for s, c, a, cum in dl_rows:
        c_w = textwrap.fill(c, width=15)
        a_w = textwrap.fill(a, width=22)
        # Split depth annotation from cumulative body
        if "  (d=" in cum:
            cum_body, d_tail = cum.rsplit("  (d=", 1)
            depth_str = f"(d={d_tail}"
        else:
            cum_body, depth_str = cum, ""
        atoms_list = [x.strip() for x in cum_body.split(" ⊓ ") if x.strip()]
        cum_w = _fmt_cum(atoms_list, depth_str)
        wrapped_rows.append((s, c_w, a_w, cum_w))

    n_rows = len(wrapped_rows)

    table = ax3.table(
        cellText=wrapped_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
        colLoc="center",
        colWidths=[0.10, 0.22, 0.26, 0.42],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.2)

    # -- Per-row height sizing ------------------------------------------
    # Distribute axes height (normalized 0..1) among header + data rows
    # proportional to each row's line count, so cells are always big enough.
    HEADER_H = 0.07
    row_line_counts = [
        max(str(cell_val).count("\n") + 1 for cell_val in row)
        for row in wrapped_rows
    ]
    total_lines = sum(row_line_counts)
    available = max(0.01, 1.0 - HEADER_H)
    per_line_h = available / max(1, total_lines)

    for (r, c_idx), cell in table.get_celld().items():
        if r == 0:
            cell.set_height(HEADER_H)
            cell.set_text_props(weight="bold", color="#111111")
            cell.set_linewidth(0.8)
        else:
            cell.set_height(per_line_h * row_line_counts[r - 1])
            cell.set_linewidth(0.6)
    # ------------------------------------------------------------------

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _draw_candidate_2(
    output_path: Path,
    g: nx.Graph,
    pos: Dict[Node, Tuple[float, float]],
    pheromone: Dict[Edge, float],
    top_paths: Sequence[Sequence[Node]],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8), dpi=220)
    widths = _edge_widths(pheromone, scale=4.8)

    all_edges = list(g.edges())
    base_widths = [widths.get(_canonical_edge(a, b), 0.8) for a, b in all_edges]
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#7F8C8D", alpha=0.55, width=base_widths)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=80, node_color="#1F7A4D", alpha=0.95)

    colors = ["#D62728", "#1F77B4", "#FF7F0E"]
    for p, c in zip(top_paths, colors):
        ed = [_canonical_edge(p[i], p[i + 1]) for i in range(len(p) - 1)]
        nx.draw_networkx_edges(g, pos, ax=ax, edgelist=ed, edge_color=c, width=3.0, alpha=0.95)

    ax.set_title("ACO Pheromone Reinforcement and Elite Paths")
    ax.axis("off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _draw_candidate_3(
    output_path: Path,
    g: nx.Graph,
    pos: Dict[Node, Tuple[float, float]],
    snapshots: Sequence[Tuple[int, Dict[Edge, float]]],
) -> None:
    cols = 4
    fig, axes = plt.subplots(1, cols, figsize=(16, 4.8), dpi=220)
    stage_names = ["t0 Init", "t1 Explore", "t2 Reinforce", "t3 Converge"]

    padded = list(snapshots)
    while len(padded) < 4:
        padded.append(padded[-1])

    for i in range(cols):
        ax = axes[i]
        gen, ph = padded[i]
        # Amplify visual contrast across stages.
        stage_scale = [1.5, 2.6, 4.2, 5.6][i]
        widths = _edge_widths(ph, scale=stage_scale)
        ed = list(g.edges())
        w = [widths.get(_canonical_edge(a, b), 0.7) for a, b in ed]

        # Base graph
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#C6CED3", alpha=0.45, width=0.7)
        nx.draw_networkx_nodes(g, pos, ax=ax, node_size=62, node_color="#A7D7C5", alpha=0.9)

        # Overlay pheromone-weighted edges
        nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#5D6D7E", alpha=0.7, width=w)

        # Highlight top reinforced edges at each stage for intuitive progression.
        top_k = [6, 8, 12, 16][i]
        top_edges = sorted(ph.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_edgelist = [(a, b) for (a, b), _ in top_edges]
        nx.draw_networkx_edges(
            g,
            pos,
            ax=ax,
            edgelist=top_edgelist,
            edge_color="#D62728",
            alpha=0.95,
            width=2.2,
        )

        # Highlight nodes touched by top edges.
        top_nodes = sorted({n for e in top_edgelist for n in e})
        nx.draw_networkx_nodes(
            g,
            pos,
            ax=ax,
            nodelist=top_nodes,
            node_size=82,
            node_color="#D62728",
            alpha=0.9,
        )

        ax.set_title(f"{stage_names[i]} (gen={gen + 1})", fontsize=10)
        ax.text(
            0.02,
            0.02,
            f"Top edges: {top_k}",
            transform=ax.transAxes,
            fontsize=8,
            color="#333333",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "#BBBBBB"},
        )
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path("output/figures/aco_candidates")
    owl = Path("ontology/chebi.owl")

    labels, nodes, edges, level, root = _build_semantic_subgraph(
        owl_path=owl,
        root_label="chemical entity",
        max_depth=8,
        max_nodes=1000,
        target_nodes=70,
    )

    result = _simulate_aco(
        nodes=nodes,
        edges=edges,
        level=level,
        root=root,
        labels=labels,
        generations=6,
        ants_per_gen=30,
        max_steps=7,
    )

    g: nx.Graph = result["graph"]
    pos = nx.spring_layout(g, seed=42, k=1.7 / math.sqrt(max(1, g.number_of_nodes())), iterations=220)

    candidate1 = out_dir / "figure_candidate_1_storyboard.png"
    candidate2 = out_dir / "figure_candidate_2_pheromone.png"
    candidate3 = out_dir / "figure_candidate_3_timeline.png"

    _draw_candidate_1(candidate1, g, pos, result["top_paths"][0], result["top_dl_rows"][0])
    _draw_candidate_2(candidate2, g, pos, result["pheromone"], result["top_paths"])
    _draw_candidate_3(candidate3, g, pos, result["snapshots"])

    print(f"[DONE] Saved: {candidate1}")
    print(f"[DONE] Saved: {candidate2}")
    print(f"[DONE] Saved: {candidate3}")


if __name__ == "__main__":
    main()
