"""
HierarchicalSearch - 계층 구조 기반 탐색 공간 엔트로피 통제

HIV(GO), Tox21(AOP-Wiki), SIDER(MedDRA)처럼 계층 구조가 깊은
온톨로지에서 상위 개념(SOC, MIE, Viral replication) 노드에 도달한
개미에게만 하위 세부 노드로 진입할 권한을 부여하여
탐색 공간의 무질서도(Entropy)를 통제한다.

핵심 매커니즘:
    1. GateKeeper: 계층의 각 레벨에 "게이트"를 설치
    2. DepthTracker: 개미가 현재 도달한 계층 깊이를 추적
    3. EntropyCap: 하위 탐색이 가능한 최대 깊이를 동적으로 제한
    4. GatePheromoneBonus: 게이트 통과 시 보너스 페로몬 부여

Usage:
    hs = HierarchicalSearch(hierarchy_config, graph)
    hs.install_gates()
    # AntPathFinder의 candidate 필터링에 통합
    filtered = hs.filter_candidates(current_node, candidates, visited_gates)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class GateNode:
    """계층 탐색의 게이트 노드 정보.

    Attributes
    ----------
    node_id : str
        게이트에 해당하는 그래프 노드 ID.
    depth : int
        계층 내 깊이 레벨 (0 = 최상위).
    concept : str
        게이트 개념명 (예: "biological_process").
    pheromone_bonus : float
        게이트 통과 시 보너스 페로몬.
    """

    __slots__ = ("node_id", "depth", "concept", "pheromone_bonus")

    def __init__(
        self,
        node_id: str,
        depth: int,
        concept: str,
        pheromone_bonus: float = 2.0,
    ) -> None:
        self.node_id = node_id
        self.depth = depth
        self.concept = concept
        self.pheromone_bonus = pheromone_bonus

    def __repr__(self) -> str:
        return (
            f"GateNode(id={self.node_id!r}, depth={self.depth}, "
            f"concept={self.concept!r})"
        )


class HierarchicalSearch:
    """계층 구조 기반 탐색 공간 엔트로피 통제기.

    Parameters
    ----------
    hierarchy_config : dict
        ``dataset_ontology_config.json``의 ``hierarchy_config`` 섹션.
        구조: {
            "gate_levels": [
                {"depth": 0, "concept": "...", "comment": "..."},
                ...
            ],
            "max_depth_without_gate": 2,
            "gate_pheromone_bonus": 3.0
        }
    graph : nx.DiGraph
        온톨로지 그래프.

    Examples
    --------
    >>> hs = HierarchicalSearch(config, G)
    >>> hs.install_gates()
    >>> candidates = hs.filter_candidates("nodeA", all_cands, visited_gates)
    """

    def __init__(
        self,
        hierarchy_config: Dict[str, Any],
        graph: nx.DiGraph,
    ) -> None:
        self.config = hierarchy_config
        self.graph = graph

        self._gate_levels = hierarchy_config.get("gate_levels", [])
        self._max_depth_without_gate = hierarchy_config.get(
            "max_depth_without_gate", 2
        )
        self._gate_bonus = hierarchy_config.get("gate_pheromone_bonus", 3.0)
        self._source_ontology = hierarchy_config.get("source_ontology", "dto")

        # 설치된 게이트 노드 레지스트리
        self._gates: Dict[int, GateNode] = {}  # depth → GateNode
        self._gate_node_ids: Set[str] = set()
        self._child_zones: Dict[str, Set[str]] = {}  # gate_id → 하위 노드 집합

        self._installed = False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 공개 API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def install_gates(self) -> int:
        """그래프에 계층 게이트를 설치한다.

        config의 gate_levels에 정의된 각 개념(concept)과 일치하는
        노드를 찾거나 새로 생성하고, 게이트 속성을 부여한다.

        Returns
        -------
        int
            설치된 게이트 수.
        """
        G = self.graph
        installed = 0

        for level_cfg in self._gate_levels:
            depth = level_cfg["depth"]
            concept = level_cfg["concept"]

            # 온톨로지에서 일치하는 노드 탐색
            gate_nid = self._find_or_create_gate_node(concept, depth)

            gate = GateNode(
                node_id=gate_nid,
                depth=depth,
                concept=concept,
                pheromone_bonus=self._gate_bonus,
            )
            self._gates[depth] = gate
            self._gate_node_ids.add(gate_nid)

            # 게이트 노드에 속성 설정
            G.nodes[gate_nid]["is_gate"] = True
            G.nodes[gate_nid]["gate_depth"] = depth
            G.nodes[gate_nid]["gate_concept"] = concept

            # 게이트 노드에 보너스 페로몬 엣지 설정
            for u, v, edata in G.edges(gate_nid, data=True):
                edata["pheromone"] = max(
                    edata.get("pheromone", 1.0),
                    self._gate_bonus,
                )
            for u, v, edata in G.in_edges(gate_nid, data=True):
                edata["pheromone"] = max(
                    edata.get("pheromone", 1.0),
                    self._gate_bonus,
                )

            installed += 1

        # 게이트 간 계단 구조 연결
        self._link_gates()

        # 각 게이트의 하위 존 계산
        self._compute_child_zones()

        self._installed = True
        logger.info(
            "HierarchicalSearch: installed %d gates (max_depth_without_gate=%d)",
            installed,
            self._max_depth_without_gate,
        )
        return installed

    def filter_candidates(
        self,
        current_node: str,
        candidates: List[str],
        visited_gates: Set[str],
    ) -> List[str]:
        """개미의 후보 노드를 계층 게이트 규칙에 따라 필터링한다.

        상위 게이트를 통과하지 않은 개미는 하위 세부 노드에
        진입할 수 없다.

        Parameters
        ----------
        current_node : str
            현재 개미 위치.
        candidates : list[str]
            이동 가능한 후보 노드 리스트.
        visited_gates : set[str]
            이 개미가 이미 통과한 게이트 노드 ID 집합.

        Returns
        -------
        list[str]
            필터링된 후보 노드 리스트.
        """
        if not self._installed or not self._gates:
            return candidates

        # 사용자가 도달한 최대 게이트 깊이
        max_visited_depth = -1
        for gate in self._gates.values():
            if gate.node_id in visited_gates:
                max_visited_depth = max(max_visited_depth, gate.depth)

        # 허용 최대 깊이 = max_visited_depth + max_depth_without_gate + 1
        allowed_max_depth = max_visited_depth + self._max_depth_without_gate + 1

        filtered = []
        for cand in candidates:
            cand_data = self.graph.nodes.get(cand, {})

            # 게이트 노드는 항상 접근 가능
            if cand in self._gate_node_ids:
                filtered.append(cand)
                continue

            # 게이트 깊이가 없는(일반) 노드는 통과
            cand_depth = cand_data.get("hierarchy_depth", -1)
            if cand_depth < 0:
                # 깊이가 미할당된 노드 - 게이트 존에 속하는지 확인
                in_gated_zone = False
                for gate_nid, zone_nodes in self._child_zones.items():
                    if cand in zone_nodes:
                        # 해당 게이트를 통과했는지 확인
                        gate_depth = self.graph.nodes[gate_nid].get("gate_depth", 0)
                        if gate_depth > max_visited_depth + 1:
                            in_gated_zone = True
                            break

                if not in_gated_zone:
                    filtered.append(cand)
                continue

            # 깊이가 할당된 노드 - 허용 깊이 내인지 확인
            if cand_depth <= allowed_max_depth:
                filtered.append(cand)

        # 필터링이 너무 공격적이면 (후보가 0개) 원래 리스트 반환
        if not filtered and candidates:
            return candidates

        return filtered

    def update_visited_gates(
        self,
        node_id: str,
        visited_gates: Set[str],
    ) -> Set[str]:
        """개미가 노드를 방문할 때 게이트 통과 여부를 업데이트한다.

        Parameters
        ----------
        node_id : str
            방문한 노드 ID.
        visited_gates : set[str]
            현재까지 통과한 게이트 집합 (in-place 업데이트됨).

        Returns
        -------
        set[str]
            업데이트된 게이트 집합.
        """
        if node_id in self._gate_node_ids:
            visited_gates.add(node_id)
        return visited_gates

    def grant_gate_bonus(
        self,
        path: List[str],
    ) -> float:
        """경로에서 게이트 통과 횟수에 따른 보너스를 계산한다.

        Parameters
        ----------
        path : list[str]
            개미의 탐색 경로.

        Returns
        -------
        float
            게이트 보너스 배율 (1.0 이상).
        """
        gates_visited = sum(1 for n in path if n in self._gate_node_ids)
        if gates_visited == 0:
            return 1.0
        # 게이트를 더 많이 통과할수록 보너스 증가 (체감 수확)
        return 1.0 + self._gate_bonus * math.log1p(gates_visited)

    def compute_entropy(self) -> float:
        """현재 탐색 공간의 엔트로피를 계산한다.

        Shannon entropy: H = -sum(p_i * log2(p_i))
        여기서 p_i는 각 게이트 zone의 크기 비율.

        Returns
        -------
        float
            탐색 공간 엔트로피 (bits).
        """
        total = sum(len(z) for z in self._child_zones.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for zone_nodes in self._child_zones.values():
            if len(zone_nodes) == 0:
                continue
            p = len(zone_nodes) / total
            entropy -= p * math.log2(p)

        return entropy

    @property
    def gates(self) -> Dict[int, GateNode]:
        """깊이별 게이트 딕셔너리."""
        return dict(self._gates)

    @property
    def gate_node_ids(self) -> Set[str]:
        """게이트 노드 ID 집합."""
        return set(self._gate_node_ids)

    def print_hierarchy(self) -> None:
        """계층 게이트 구조를 출력한다."""
        print("=" * 55)
        print(" Hierarchical Search Gate Structure")
        print("=" * 55)
        for depth in sorted(self._gates.keys()):
            gate = self._gates[depth]
            zone_size = len(self._child_zones.get(gate.node_id, set()))
            print(
                f"  L{depth}: {gate.concept:30s} "
                f"(node={gate.node_id}, zone={zone_size} nodes)"
            )
        entropy = self.compute_entropy()
        print(f"  Entropy: {entropy:.4f} bits")
        print("=" * 55)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 내부 구현
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _find_or_create_gate_node(self, concept: str, depth: int) -> str:
        """개념명과 일치하는 노드를 찾거나 새로 생성한다."""
        G = self.graph
        concept_lower = concept.lower()

        # 1) 정확 일치 또는 부분 일치 검색
        for nid, ndata in G.nodes(data=True):
            label = ndata.get("label", nid).lower()
            if concept_lower == label or concept_lower in label:
                return nid

        # 2) 없으면 새 게이트 노드 생성
        gate_nid = f"gate:{concept}"
        G.add_node(
            gate_nid,
            label=concept,
            uri=f"urn:hierarchy:{concept}",
        )

        # hub 노드에 연결
        hub = None
        for candidate in ["Thing", "thing", "owl:Thing"]:
            if candidate in G:
                hub = candidate
                break
        if hub is None and G.number_of_nodes() > 1:
            hub = max(
                (n for n in G.nodes if n != gate_nid),
                key=lambda n: G.degree(n),
            )

        if hub and hub in G:
            G.add_edge(
                hub, gate_nid,
                predicate="hasHierarchyGate",
                pheromone=self._gate_bonus,
            )

        return gate_nid

    def _link_gates(self) -> None:
        """게이트 간 계단식 연결을 생성한다 (depth 0 → 1 → 2 ...)."""
        G = self.graph
        sorted_depths = sorted(self._gates.keys())

        for i in range(len(sorted_depths) - 1):
            d_curr = sorted_depths[i]
            d_next = sorted_depths[i + 1]
            gate_curr = self._gates[d_curr]
            gate_next = self._gates[d_next]

            if not G.has_edge(gate_curr.node_id, gate_next.node_id):
                G.add_edge(
                    gate_curr.node_id,
                    gate_next.node_id,
                    predicate="nextHierarchyLevel",
                    pheromone=self._gate_bonus * 0.8,
                )

    def _compute_child_zones(self) -> None:
        """각 게이트 노드로부터 도달 가능한 하위 노드 집합을 계산한다."""
        G = self.graph

        for gate in self._gates.values():
            # BFS로 게이트에서 도달 가능한 모든 하위 노드 수집
            zone: Set[str] = set()
            visited: Set[str] = {gate.node_id}
            queue = [gate.node_id]

            max_bfs_depth = 5  # 무한 확장 방지
            depth_map = {gate.node_id: 0}

            while queue:
                current = queue.pop(0)
                d = depth_map[current]
                if d >= max_bfs_depth:
                    continue

                for neighbor in set(G.successors(current)) | set(G.predecessors(current)):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        zone.add(neighbor)
                        depth_map[neighbor] = d + 1
                        queue.append(neighbor)

                        # 하위 노드에 hierarchy_depth 할당
                        if "hierarchy_depth" not in G.nodes[neighbor]:
                            G.nodes[neighbor]["hierarchy_depth"] = (
                                gate.depth * max_bfs_depth + d + 1
                            )

            self._child_zones[gate.node_id] = zone

        logger.debug(
            "Child zones computed: %s",
            {g: len(z) for g, z in self._child_zones.items()},
        )
