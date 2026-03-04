"""
Step 3: 개미의 확률적 이동 시뮬레이션 (AntPathFinder)

구축된 온톨로지 그래프 위에서 개미 한 마리가 ACO 확률 규칙에 따라
확률적으로 이동하며, is_feature == True 노드 방향으로 휴리스틱(η)
가중치를 적용한다.

이동 확률:
    P(i → j) = [τ_ij^α · η_ij^β] / Σ_k [τ_ik^α · η_ik^β]

여기서
    τ = pheromone (페로몬)
    η = heuristic  (is_feature 노드에 대한 선호도)
    α = pheromone 지수 (기본 1.0)
    β = heuristic 지수 (기본 2.0)
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Set

import networkx as nx

logger = logging.getLogger(__name__)


class AntPathFinder:
    """개미 한 마리의 확률적 그래프 탐색을 시뮬레이션한다.

    Parameters
    ----------
    graph : nx.DiGraph
        OWLGraphBuilder가 생성하고, SMILESOntologyBridge가 is_feature
        속성을 부여한 온톨로지 DiGraph.
    alpha : float
        페로몬(τ) 지수. 기본 1.0.
    beta : float
        휴리스틱(η) 지수. 기본 2.0.
    feature_heuristic : float
        is_feature==True 노드로 이동할 때의 휴리스틱 상수 η. 기본 3.0.
    default_heuristic : float
        일반 노드로 이동할 때의 기본 η. 기본 1.0.
    max_steps : int
        한 번의 탐색에서 최대 이동 횟수. 기본 50.
    seed : int | None
        재현성을 위한 난수 시드.

    Examples
    --------
    >>> ant = AntPathFinder(G, alpha=1.0, beta=2.0, feature_heuristic=3.0)
    >>> path = ant.explore(start_node="SomeClass")
    >>> print(path)
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        alpha: float = 1.0,
        beta: float = 2.0,
        feature_heuristic: float = 3.0,
        default_heuristic: float = 1.0,
        max_steps: int = 50,
        seed: Optional[int] = None,
    ) -> None:
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.feature_heuristic = feature_heuristic
        self.default_heuristic = default_heuristic
        self.max_steps = max_steps

        self._rng = random.Random(seed)

    # ── 공개 API ───────────────────────────────────────────────

    def explore(
        self,
        start_node: Optional[str] = None,
        *,
        allow_revisit: bool = False,
    ) -> List[str]:
        """개미가 start_node에서 출발하여 확률적으로 이동하고,
        지나간 노드 리스트를 반환한다.

        Parameters
        ----------
        start_node : str | None
            출발 노드 ID. None이면 임의의 노드에서 시작.
        allow_revisit : bool
            이미 방문한 노드를 재방문할 수 있는지 여부.

        Returns
        -------
        list[str]
            방문한 노드 ID 순서 리스트.
        """
        G = self.graph

        if start_node is None:
            start_node = self._rng.choice(list(G.nodes))
        elif start_node not in G:
            raise ValueError(f"Node '{start_node}' not in graph.")

        path: List[str] = [start_node]
        visited: Set[str] = {start_node}

        current = start_node
        for step in range(self.max_steps):
            candidates = self._get_candidates(current, visited, allow_revisit)
            if not candidates:
                logger.debug(
                    "Ant stuck at '%s' after %d steps — exploring ended.",
                    current,
                    step,
                )
                break

            next_node = self._select_next(current, candidates)
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        return path

    def explore_multi(
        self,
        n_ants: int = 10,
        start_node: Optional[str] = None,
        *,
        allow_revisit: bool = False,
    ) -> List[List[str]]:
        """여러 개미를 동시에 출발시키고 모든 경로를 반환한다.

        Parameters
        ----------
        n_ants : int
            개미 수.
        start_node : str | None
            공통 출발 노드. None이면 각 개미가 랜덤 시작.

        Returns
        -------
        list[list[str]]
        """
        return [
            self.explore(start_node, allow_revisit=allow_revisit)
            for _ in range(n_ants)
        ]

    def update_pheromones(
        self,
        path: List[str],
        deposit: float = 1.0,
        evaporation: float = 0.1,
    ) -> None:
        """경로를 따라 페로몬을 갱신한다.

        1) 전역 증발: 모든 엣지의 τ *= (1 - evaporation)
        2) 경로 강화: 경로 상 엣지에 deposit 추가.

        Parameters
        ----------
        path : list[str]
        deposit : float
        evaporation : float
        """
        G = self.graph

        # 1) 전역 증발
        for u, v, edata in G.edges(data=True):
            edata["pheromone"] = edata.get("pheromone", 1.0) * (1 - evaporation)
            # 페로몬 하한 (0에 수렴 방지)
            if edata["pheromone"] < 1e-6:
                edata["pheromone"] = 1e-6

        # 2) 경로 강화
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if G.has_edge(u, v):
                G.edges[u, v]["pheromone"] += deposit
            # 역방향 엣지가 있을 수도 있음
            if G.has_edge(v, u):
                G.edges[v, u]["pheromone"] += deposit

    # ── 내부 구현 ──────────────────────────────────────────────

    def _get_candidates(
        self,
        current: str,
        visited: Set[str],
        allow_revisit: bool,
    ) -> List[str]:
        """현재 노드에서 이동 가능한 후보 노드 리스트를 반환한다.

        DiGraph이므로 successors (나가는 방향) + predecessors (들어오는
        방향) 모두 고려하여 양방향 탐색을 허용한다.
        """
        G = self.graph
        neighbors = set(G.successors(current)) | set(G.predecessors(current))
        if not allow_revisit:
            neighbors -= visited
        return list(neighbors)

    def _select_next(self, current: str, candidates: List[str]) -> str:
        """ACO 확률 규칙에 따라 다음 노드를 선택한다.

        P(i → j) = [τ_ij^α · η_j^β] / Σ_k [τ_ik^α · η_k^β]
        """
        G = self.graph
        weights: List[float] = []

        for cand in candidates:
            # 페로몬 τ: 엣지 속성에서 가져옴 (양방향 중 존재하는 쪽)
            if G.has_edge(current, cand):
                tau = G.edges[current, cand].get("pheromone", 1.0)
            elif G.has_edge(cand, current):
                tau = G.edges[cand, current].get("pheromone", 1.0)
            else:
                tau = 1.0

            # 휴리스틱 η: is_feature 노드에 가중치 부여
            is_feat = G.nodes[cand].get("is_feature", False)
            eta = self.feature_heuristic if is_feat else self.default_heuristic

            w = (tau ** self.alpha) * (eta ** self.beta)
            weights.append(w)

        total = sum(weights)
        if total == 0:
            # 모든 가중치가 0이면 균등 선택
            return self._rng.choice(candidates)

        # 확률 기반 룰렛 선택
        probabilities = [w / total for w in weights]
        return self._rng.choices(candidates, weights=probabilities, k=1)[0]

    # ── 유틸리티 ───────────────────────────────────────────────

    def path_summary(self, path: List[str]) -> Dict[str, Any]:
        """탐색 경로 요약 정보를 반환한다."""
        G = self.graph
        feature_nodes = [n for n in path if G.nodes[n].get("is_feature")]
        total_pheromone = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if G.has_edge(u, v):
                total_pheromone += G.edges[u, v].get("pheromone", 0)
            elif G.has_edge(v, u):
                total_pheromone += G.edges[v, u].get("pheromone", 0)

        return {
            "length": len(path),
            "unique_nodes": len(set(path)),
            "feature_nodes_visited": feature_nodes,
            "feature_count": len(feature_nodes),
            "total_pheromone": round(total_pheromone, 4),
        }


# ── CLI 실행 (데모) ───────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    from src.aco.owl_graph_builder import OWLGraphBuilder
    from src.aco.smiles_ontology_bridge import SMILESOntologyBridge

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    owl_file = sys.argv[1] if len(sys.argv) > 1 else "ontology/DTO.owl"

    # ── Step 1: 그래프 빌드
    builder = OWLGraphBuilder(owl_file)
    G = builder.build()
    builder.print_summary()

    # ── Step 2: Feature 브릿지
    bridge = SMILESOntologyBridge(G, match_threshold=65)
    sample_features = [
        "molecular_weight", "logp", "tpsa", "num_hbd", "num_hba",
        "num_rotatable_bonds", "num_rings", "num_aromatic_rings",
    ]
    mapping = bridge.auto_map(sample_features)
    unmapped = [f for f in sample_features if f not in mapping]
    if unmapped:
        bridge.create_feature_nodes(unmapped)
    bridge.print_mapping()

    # ── Step 3: 개미 시뮬레이션
    print("\n" + "=" * 60)
    print("ANT PATH SIMULATION")
    print("=" * 60)

    ant = AntPathFinder(
        G,
        alpha=1.0,
        beta=2.0,
        feature_heuristic=3.0,
        default_heuristic=1.0,
        max_steps=30,
        seed=42,
    )

    path = ant.explore()
    summary = ant.path_summary(path)
    print(f"\nPath length: {summary['length']}")
    print(f"Unique nodes: {summary['unique_nodes']}")
    print(f"Feature nodes visited: {summary['feature_count']}")
    print(f"Total pheromone: {summary['total_pheromone']}")
    print(f"\nPath (first 10):")
    for i, node in enumerate(path[:10]):
        marker = " ★" if G.nodes[node].get("is_feature") else ""
        print(f"  [{i}] {node}{marker}")

    # 페로몬 업데이트 후 재탐색
    ant.update_pheromones(path, deposit=2.0, evaporation=0.05)
    path2 = ant.explore(start_node=path[0])
    summary2 = ant.path_summary(path2)
    print(f"\n[After pheromone update] Path length: {summary2['length']}, "
          f"Features visited: {summary2['feature_count']}")
