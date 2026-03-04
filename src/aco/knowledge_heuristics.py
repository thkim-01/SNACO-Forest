"""
KnowledgeHeuristics - 추상적 도메인 지식을 수치적 휴리스틱으로 변환

3가지 전략 패턴을 구현:
    1. PhysicochemicalPriorBounds (BBBP)
       - ChEBI 유래 TPSA, LogP 등의 임계범위를 Prior Bounds로 설정
       - 개미의 분할 지점 탐색을 도메인 지식으로 제한
    2. StaticPheromone (ClinTox)
       - FDA EDT 6단계 질문 + Cramer 3단계 분류를
         트리 상위 노드에 고정 페로몬으로 강제 할당
    3. BindingPocketHeuristic (BACE)
       - PDB S1/S3 포켓 점유, Asp32/Asp228 상호작용을
         이진(Binary) 특성으로 변환하여 eta(η) 계산에 포함

Usage:
    heuristics = KnowledgeHeuristics(config, graph)
    heuristics.apply("bbbp", feature_df)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. PhysicochemicalPriorBounds (BBBP 등)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class PhysicochemicalPriorBounds:
    """SMILES 특성의 분할 탐색 범위를 도메인 지식 기반 Prior Bounds로 제한.

    ChEBI에서 정의된 TPSA (<90), LogP (-0.5~5.0), MW (<500) 등의
    물리화학적 임계값이 개미가 분할 지점(split point)을 찾을 때
    탐색 범위를 제한하는 Prior 역할을 한다.

    Parameters
    ----------
    prior_bounds : dict
        ``{"feature_name": {"low": float, "high": float}}``
    graph : nx.DiGraph
        온톨로지 그래프 (feature 노드에 prior_low/prior_high 설정).

    Examples
    --------
    >>> bounds = PhysicochemicalPriorBounds(
    ...     {"tpsa": {"low": 0, "high": 90}}, G
    ... )
    >>> bounds.apply(feature_df)
    """

    def __init__(
        self,
        prior_bounds: Dict[str, Dict[str, float]],
        graph: nx.DiGraph,
    ) -> None:
        self.prior_bounds = prior_bounds
        self.graph = graph
        self._applied = False

    def apply(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """Prior Bounds를 그래프 노드 속성에 설정하고,
        feature_df의 값을 bounds 범위로 클리핑한 DataFrame을 반환.

        Parameters
        ----------
        feature_df : pd.DataFrame
            원본 특성 데이터프레임.

        Returns
        -------
        pd.DataFrame
            Prior Bounds가 적용된 클리핑 데이터프레임.
        """
        clipped = feature_df.copy()
        applied_count = 0

        for feat_name, bounds in self.prior_bounds.items():
            low = bounds.get("low", float("-inf"))
            high = bounds.get("high", float("inf"))

            # 그래프 노드에 prior bounds 속성 설정
            for nid, ndata in self.graph.nodes(data=True):
                fkey = ndata.get("feature_key", "")
                if fkey == feat_name or ndata.get("label", "") == feat_name:
                    ndata["prior_low"] = low
                    ndata["prior_high"] = high
                    ndata["has_prior"] = True
                    applied_count += 1
                    break

            # DataFrame 값 클리핑
            if feat_name in clipped.columns:
                clipped[feat_name] = clipped[feat_name].clip(lower=low, upper=high)

        self._applied = True
        logger.info(
            "PhysicochemicalPriorBounds applied: %d/%d features bound",
            applied_count,
            len(self.prior_bounds),
        )
        return clipped

    def get_bounds_for_feature(
        self,
        feature_name: str,
    ) -> Optional[Tuple[float, float]]:
        """특정 feature의 prior bounds를 반환."""
        if feature_name in self.prior_bounds:
            b = self.prior_bounds[feature_name]
            return (b.get("low", float("-inf")), b.get("high", float("inf")))
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. StaticPheromone (ClinTox)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class StaticPheromone:
    """FDA EDT 6단계 질문과 Cramer 3단계 분류를 트리 상위 노드에
    고정 페로몬(static pheromone)으로 강제 할당한다.

    이 고정 페로몬은 증발(evaporation)에 영향받지 않으며,
    개미가 독성 관련 의사결정 경로를 우선 탐색하도록 유도한다.

    Parameters
    ----------
    static_config : dict
        ``dataset_ontology_config.json``의 ``static_pheromone`` 섹션.
    graph : nx.DiGraph
        온톨로지 그래프.
    pheromone_multiplier : float
        고정 페로몬 배율. 기본 5.0.

    Examples
    --------
    >>> sp = StaticPheromone(config, G, pheromone_multiplier=5.0)
    >>> sp.apply(feature_df)
    """

    def __init__(
        self,
        static_config: Dict[str, Any],
        graph: nx.DiGraph,
        pheromone_multiplier: float = 5.0,
    ) -> None:
        self.static_config = static_config
        self.graph = graph
        self.pheromone_multiplier = pheromone_multiplier

        self._edt_questions = static_config.get("fda_edt_questions", [])
        self._cramer_classes = static_config.get("cramer_classes", {})
        self._static_nodes: List[str] = []  # 고정 페로몬 적용된 노드

    def apply(self, feature_df: pd.DataFrame) -> None:
        """고정 페로몬 노드를 그래프에 삽입하고 페로몬을 설정한다.

        Parameters
        ----------
        feature_df : pd.DataFrame
            특성 데이터프레임 (Cramer 분류 통계 추출에 사용).
        """
        G = self.graph
        node_count_before = G.number_of_nodes()

        # ── FDA EDT 6단계 질문 노드 생성 ──
        edt_nodes = self._create_edt_nodes()

        # ── Cramer 3단계 분류 노드 생성 ──
        cramer_nodes = self._create_cramer_nodes(feature_df)

        self._static_nodes = edt_nodes + cramer_nodes

        logger.info(
            "StaticPheromone applied: %d EDT nodes + %d Cramer nodes "
            "(graph %d → %d nodes)",
            len(edt_nodes),
            len(cramer_nodes),
            node_count_before,
            G.number_of_nodes(),
        )

    def get_static_nodes(self) -> List[str]:
        """고정 페로몬이 적용된 노드 ID 리스트."""
        return list(self._static_nodes)

    def is_static_node(self, node_id: str) -> bool:
        """해당 노드가 고정 페로몬 노드인지 확인."""
        return node_id in self._static_nodes

    def protect_from_evaporation(self) -> None:
        """고정 페로몬 노드의 엣지를 증발에서 보호한다.

        매 세대(generation) 종료 후 호출하면 고정 페로몬을 복원한다.
        """
        G = self.graph
        restored = 0
        for nid in self._static_nodes:
            if nid not in G:
                continue
            target_pher = self.pheromone_multiplier
            for u, v, edata in G.edges(nid, data=True):
                if edata.get("pheromone", 1.0) < target_pher:
                    edata["pheromone"] = target_pher
                    edata["static"] = True
                    restored += 1
            for u, v, edata in G.in_edges(nid, data=True):
                if edata.get("pheromone", 1.0) < target_pher:
                    edata["pheromone"] = target_pher
                    edata["static"] = True
                    restored += 1

        if restored:
            logger.debug("StaticPheromone restored %d edges", restored)

    def _create_edt_nodes(self) -> List[str]:
        """FDA EDT 6단계 질문을 그래프 노드로 생성한다."""
        G = self.graph
        created: List[str] = []

        # hub 노드 탐지
        hub = self._find_hub_node()

        for q in self._edt_questions:
            level = q["level"]
            feature = q["feature"]
            threshold = q["threshold"]
            question = q.get("question", f"EDT_L{level}")

            node_id = f"edt:L{level}_{feature}"
            G.add_node(
                node_id,
                label=f"EDT-L{level}: {question}",
                uri=f"urn:fda_edt:level{level}",
                is_feature=True,
                feature_key=feature,
                prior_low=float("-inf"),
                prior_high=threshold,
                has_prior=True,
                static_pheromone=True,
                edt_level=level,
            )

            # hub에 고정 페로몬 연결
            if hub and hub in G:
                pher = self.pheromone_multiplier * (1.0 + 0.5 * (5 - level))
                G.add_edge(
                    hub, node_id,
                    predicate="hasEDTQuestion",
                    pheromone=pher,
                    static=True,
                )

            # 이전 레벨 노드에 연결 (계단식 구조)
            if level > 0:
                prev_node = f"edt:L{level - 1}_{self._edt_questions[level - 1]['feature']}"
                if prev_node in G:
                    G.add_edge(
                        prev_node, node_id,
                        predicate="nextEDTLevel",
                        pheromone=self.pheromone_multiplier,
                        static=True,
                    )

            created.append(node_id)

        return created

    def _create_cramer_nodes(self, feature_df: pd.DataFrame) -> List[str]:
        """Cramer Classification 3단계를 그래프 노드로 생성한다."""
        G = self.graph
        created: List[str] = []

        hub = self._find_hub_node()

        cramer_levels = {
            "class_I_low": {
                "label": "Cramer I (Low Concern)",
                "pheromone_factor": 1.0,
            },
            "class_II_medium": {
                "label": "Cramer II (Medium Concern)",
                "pheromone_factor": 1.5,
            },
            "class_III_high": {
                "label": "Cramer III (High Concern)",
                "pheromone_factor": 2.0,
            },
        }

        prev_node = None
        for class_key, level_info in cramer_levels.items():
            class_cfg = self._cramer_classes.get(class_key, {})
            node_id = f"cramer:{class_key}"

            G.add_node(
                node_id,
                label=level_info["label"],
                uri=f"urn:cramer:{class_key}",
                static_pheromone=True,
                cramer_class=class_key,
            )

            # hub에 연결
            if hub and hub in G:
                pher = self.pheromone_multiplier * level_info["pheromone_factor"]
                G.add_edge(
                    hub, node_id,
                    predicate="hasCramerClass",
                    pheromone=pher,
                    static=True,
                )

            # Cramer 분류 임계값 조건을 feature 노드에 연결
            for param, value in class_cfg.items():
                if param.startswith("_") or param == "comment":
                    continue
                # 파라미터를 feature 참조 노드로 연결
                feat_name = param.replace("_max", "").replace("_min", "")
                feat_node = self._find_feature_node(feat_name)
                if feat_node and feat_node in G:
                    if not G.has_edge(node_id, feat_node):
                        G.add_edge(
                            node_id, feat_node,
                            predicate="cramerCriterion",
                            pheromone=self.pheromone_multiplier,
                            static=True,
                        )

            if prev_node:
                G.add_edge(
                    prev_node, node_id,
                    predicate="nextCramerLevel",
                    pheromone=self.pheromone_multiplier * 0.8,
                    static=True,
                )

            prev_node = node_id
            created.append(node_id)

        return created

    def _find_hub_node(self) -> Optional[str]:
        """그래프에서 최상위 hub 노드를 찾는다."""
        G = self.graph
        for candidate in ["Thing", "thing", "owl:Thing"]:
            if candidate in G:
                return candidate
        if G.number_of_nodes() > 0:
            return max(G.nodes, key=lambda n: G.degree(n))
        return None

    def _find_feature_node(self, feature_name: str) -> Optional[str]:
        """feature_key가 일치하는 노드를 찾는다."""
        for nid, ndata in self.graph.nodes(data=True):
            fkey = ndata.get("feature_key", "")
            if fkey == feature_name or ndata.get("label", "") == feature_name:
                return nid
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. BindingPocketHeuristic (BACE)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BindingPocketHeuristic:
    """PDB 구조에서 추출된 바인딩 포켓 정보를 이진 특성으로 변환하여
    개미의 휴리스틱(eta) 계산에 포함시킨다.

    BACE-1 데이터셋의 경우:
    - S1 포켓: 소수성 방향족 고리가 점유하는지 (num_aromatic_rings, logp)
    - S3 포켓: H-bond acceptor가 점유하는지 (num_hba, tpsa)
    - Asp32: 촉매 잔기와 H-bond (num_hbd, num_nitrogens)
    - Asp228: 촉매 잔기와 상호작용 (num_oxygens, num_hba)

    이 정보는 rdkit/PDB 없이 SMILES 디스크립터의 프록시 특성으로 근사한다.

    Parameters
    ----------
    pocket_config : dict
        ``dataset_ontology_config.json``의 ``pocket_features`` 섹션.
    graph : nx.DiGraph
        온톨로지 그래프.

    Examples
    --------
    >>> pocket = BindingPocketHeuristic(config, G)
    >>> feature_df = pocket.apply(feature_df)
    """

    def __init__(
        self,
        pocket_config: Dict[str, Any],
        graph: nx.DiGraph,
    ) -> None:
        self.pocket_config = pocket_config
        self.graph = graph
        self._binary_features: List[str] = []

    def apply(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """바인딩 포켓 이진 특성을 DataFrame에 추가하고
        그래프에 feature 노드를 생성한다.

        Parameters
        ----------
        feature_df : pd.DataFrame
            원본 특성 데이터프레임.

        Returns
        -------
        pd.DataFrame
            이진 포켓 특성이 추가된 데이터프레임.
        """
        result_df = feature_df.copy()

        for pocket_name, pocket_info in self.pocket_config.items():
            proxy_features = pocket_info.get("proxy_features", [])
            binary_col = f"pocket_{pocket_name}"

            # 프록시 특성으로부터 이진 점유 여부를 추정
            if len(proxy_features) == 0:
                continue

            # 프록시 특성의 중앙값 이상이면 1 (점유), 아니면 0
            present = proxy_features[0] in feature_df.columns
            if present:
                proxy_vals = []
                for pf in proxy_features:
                    if pf in feature_df.columns:
                        median_val = feature_df[pf].median()
                        proxy_vals.append(
                            (feature_df[pf] >= median_val).astype(float)
                        )

                if proxy_vals:
                    # 복수 프록시의 평균이 0.5 이상이면 점유
                    combined = sum(proxy_vals) / len(proxy_vals)
                    result_df[binary_col] = (combined >= 0.5).astype(int)
                else:
                    result_df[binary_col] = 0
            else:
                result_df[binary_col] = 0

            self._binary_features.append(binary_col)

            # 그래프에 포켓 feature 노드 생성
            self._create_pocket_node(pocket_name, binary_col)

        logger.info(
            "BindingPocketHeuristic applied: %d binary features added (%s)",
            len(self._binary_features),
            ", ".join(self._binary_features),
        )
        return result_df

    def get_binary_features(self) -> List[str]:
        """생성된 이진 포켓 특성명 리스트."""
        return list(self._binary_features)

    def compute_pocket_eta(
        self,
        node_id: str,
        base_eta: float = 1.0,
    ) -> float:
        """포켓 관련 feature 노드의 휴리스틱(eta)을 강화한다.

        Parameters
        ----------
        node_id : str
            그래프 노드 ID.
        base_eta : float
            기본 휴리스틱 값.

        Returns
        -------
        float
            조정된 eta 값 (포켓 노드이면 base_eta * 3.0).
        """
        ndata = self.graph.nodes.get(node_id, {})
        if ndata.get("is_pocket_feature", False):
            return base_eta * 3.0
        return base_eta

    def _create_pocket_node(self, pocket_name: str, binary_col: str) -> None:
        """바인딩 포켓 feature 노드를 그래프에 추가한다."""
        G = self.graph
        node_id = f"pocket:{pocket_name}"

        G.add_node(
            node_id,
            label=f"Pocket_{pocket_name}",
            uri=f"urn:pdb:pocket:{pocket_name}",
            is_feature=True,
            feature_key=binary_col,
            is_pocket_feature=True,
        )

        # hub 노드에 연결
        hub = None
        for candidate in ["Thing", "thing", "owl:Thing"]:
            if candidate in G:
                hub = candidate
                break
        if hub is None and G.number_of_nodes() > 0:
            hub = max(G.nodes, key=lambda n: G.degree(n))

        if hub and hub in G:
            G.add_edge(
                hub, node_id,
                predicate="hasBindingPocket",
                pheromone=2.0,
            )

        # 관련 프록시 feature 노드에도 연결
        proxy_features = self.pocket_config[pocket_name].get("proxy_features", [])
        for pf in proxy_features:
            for nid, ndata in G.nodes(data=True):
                if ndata.get("feature_key") == pf:
                    if not G.has_edge(node_id, nid):
                        G.add_edge(
                            node_id, nid,
                            predicate="proxiedBy",
                            pheromone=1.5,
                        )
                    break


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 통합 인터페이스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class KnowledgeHeuristics:
    """데이터셋-특이적 도메인 지식 휴리스틱 통합 관리자.

    config.json의 heuristic_profile에 따라 적절한 전략을 자동 선택한다.

    Parameters
    ----------
    dataset_config : dict
        데이터셋의 config 딕셔너리.
    graph : nx.DiGraph
        온톨로지 그래프.

    Examples
    --------
    >>> kh = KnowledgeHeuristics(ds_config, G)
    >>> feature_df = kh.apply(feature_df)
    """

    def __init__(
        self,
        dataset_config: Dict[str, Any],
        graph: nx.DiGraph,
    ) -> None:
        self.dataset_config = dataset_config
        self.graph = graph
        self.profile = dataset_config.get("heuristic_profile", "default")

        # 전략 인스턴스
        self._prior_bounds: Optional[PhysicochemicalPriorBounds] = None
        self._static_pheromone: Optional[StaticPheromone] = None
        self._binding_pocket: Optional[BindingPocketHeuristic] = None

        self._initialize_strategies()

    def _initialize_strategies(self) -> None:
        """heuristic_profile에 따라 전략을 초기화한다."""
        if self.profile == "physicochemical_prior":
            bounds = self.dataset_config.get("prior_bounds", {})
            self._prior_bounds = PhysicochemicalPriorBounds(bounds, self.graph)

        elif self.profile == "fda_cramer_static":
            static_cfg = self.dataset_config.get("static_pheromone", {})
            multiplier = static_cfg.get("pheromone_multiplier", 5.0)
            self._static_pheromone = StaticPheromone(
                static_cfg, self.graph, pheromone_multiplier=multiplier
            )

        elif self.profile == "binding_pocket":
            pocket_cfg = self.dataset_config.get("pocket_features", {})
            self._binding_pocket = BindingPocketHeuristic(pocket_cfg, self.graph)

        # hierarchical 프로파일은 HierarchicalSearch에서 처리
        logger.info("KnowledgeHeuristics initialized: profile='%s'", self.profile)

    def apply(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """적절한 휴리스틱 전략을 적용한다.

        Parameters
        ----------
        feature_df : pd.DataFrame
            원본 특성 데이터프레임.

        Returns
        -------
        pd.DataFrame
            휴리스틱이 적용된 데이터프레임.
        """
        result_df = feature_df

        if self._prior_bounds is not None:
            result_df = self._prior_bounds.apply(result_df)

        if self._static_pheromone is not None:
            self._static_pheromone.apply(result_df)

        if self._binding_pocket is not None:
            result_df = self._binding_pocket.apply(result_df)

        return result_df

    def protect_static_pheromone(self) -> None:
        """세대 종료 후 고정 페로몬을 복원한다."""
        if self._static_pheromone is not None:
            self._static_pheromone.protect_from_evaporation()

    def get_eta_modifier(self, node_id: str, base_eta: float = 1.0) -> float:
        """노드의 도메인 지식 기반 eta 수정값을 반환한다.

        Parameters
        ----------
        node_id : str
            그래프 노드 ID.
        base_eta : float
            기본 eta 값.

        Returns
        -------
        float
            수정된 eta 값.
        """
        eta = base_eta

        # Prior bounds가 있는 feature 노드에 보너스
        ndata = self.graph.nodes.get(node_id, {})
        if ndata.get("has_prior", False):
            eta *= 1.5

        # 고정 페로몬 노드에 보너스
        if ndata.get("static_pheromone", False):
            eta *= 2.0

        # 포켓 feature 노드 보너스
        if self._binding_pocket is not None:
            eta = self._binding_pocket.compute_pocket_eta(node_id, eta)

        return eta

    @property
    def has_static_pheromone(self) -> bool:
        """StaticPheromone 전략이 활성화되었는지."""
        return self._static_pheromone is not None

    @property
    def has_prior_bounds(self) -> bool:
        """PhysicochemicalPriorBounds 전략이 활성화되었는지."""
        return self._prior_bounds is not None

    @property
    def has_binding_pocket(self) -> bool:
        """BindingPocketHeuristic 전략이 활성화되었는지."""
        return self._binding_pocket is not None
