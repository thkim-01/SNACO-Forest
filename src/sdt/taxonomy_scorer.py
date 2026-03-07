"""
Taxonomy Scorer: 온톨로지 계층 구조 기반 분할 품질 평가

1. Taxonomic Informativeness (TI)
   - 온톨로지 계층(subClassOf)에서 각 개념의 Information Content(IC)를 계산
   - ATI (Average Taxonomic Informativeness): 분할에 참여한 속성들의 평균 TI

2. Penalized Information Gain (PIG)
   PIG(S) = IG(S) × log(1 + α × ATI)
   - IG(S): 기본 정보 이득
   - ATI: 평균 계층적 정보량
   - α: 계층 구조 중요도 하이퍼파라미터

3. Semantic Similarity (Wu-Palmer)
   Sim(a, b) = 2 * depth(LCA) / (depth(a) + depth(b))
   - LCA: Lowest Common Ancestor
   - 분할된 부분집합의 평균 시맨틱 유사도를 최대화
"""

from __future__ import annotations

import math
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from functools import lru_cache

logger = logging.getLogger(__name__)


class TaxonomyScorer:
    """온톨로지 계층 구조에서 TI/PIG/Semantic Similarity를 계산한다.

    Parameters
    ----------
    ontology : owlready2.Ontology | None
        owlready2로 로드된 온톨로지 객체.
        None이면 PIG/SemanticSim은 기본 IG로 폴백한다.
    alpha : float
        PIG에서 ATI의 가중치. 기본 1.0.
    """

    def __init__(
        self,
        ontology=None,
        alpha: float = 1.0,
    ) -> None:
        self.onto = ontology
        self.alpha = alpha

        # 계층 구조 캐시
        self._depth_cache: Dict[Any, int] = {}
        self._ic_cache: Dict[Any, float] = {}
        self._lca_cache: Dict[Tuple, Any] = {}
        self._ancestor_cache: Dict[Any, Set] = {}
        self._total_concepts: int = 0
        self._concept_descendants_count: Dict[Any, int] = {}

        if ontology is not None:
            self._build_hierarchy()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 계층 구조 초기화
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_hierarchy(self) -> None:
        """온톨로지에서 계층 정보(depth, IC)를 사전 계산한다."""
        if self.onto is None:
            return

        try:
            all_classes = list(self.onto.classes())
        except Exception as e:
            logger.warning("Failed to enumerate ontology classes: %s", e)
            return

        self._total_concepts = max(len(all_classes), 1)

        # 1) 각 클래스의 depth 계산 (Thing으로부터의 최대 거리)
        for cls in all_classes:
            self._compute_depth(cls)

        # 2) 각 클래스의 descendants 수 → Information Content 계산
        for cls in all_classes:
            desc = self._get_descendants(cls)
            self._concept_descendants_count[cls] = len(desc)

        # 3) IC 사전 계산
        for cls in all_classes:
            self._ic_cache[cls] = self._compute_ic(cls)

        logger.info(
            "TaxonomyScorer: %d concepts indexed, max_depth=%d",
            self._total_concepts,
            max(self._depth_cache.values()) if self._depth_cache else 0,
        )

    def _compute_depth(self, cls) -> int:
        """클래스의 depth를 계산한다 (Thing = 0)."""
        if cls in self._depth_cache:
            return self._depth_cache[cls]

        name = getattr(cls, "name", str(cls))
        if name == "Thing" or not hasattr(cls, "is_a"):
            self._depth_cache[cls] = 0
            return 0

        parents = []
        try:
            for parent in cls.is_a:
                if hasattr(parent, "name") and hasattr(parent, "is_a"):
                    parents.append(parent)
        except Exception:
            pass

        if not parents:
            self._depth_cache[cls] = 1
            return 1

        max_parent_depth = 0
        for parent in parents:
            pd = self._compute_depth(parent)
            if pd > max_parent_depth:
                max_parent_depth = pd

        depth = max_parent_depth + 1
        self._depth_cache[cls] = depth
        return depth

    def _get_ancestors(self, cls) -> Set:
        """클래스의 모든 ancestor를 반환한다 (자기 자신 포함)."""
        if cls in self._ancestor_cache:
            return self._ancestor_cache[cls]

        ancestors: Set = {cls}
        try:
            for anc in cls.ancestors():
                if hasattr(anc, "name"):
                    ancestors.add(anc)
        except Exception:
            pass

        self._ancestor_cache[cls] = ancestors
        return ancestors

    def _get_descendants(self, cls) -> Set:
        """클래스의 모든 descendant를 반환한다 (자기 자신 포함)."""
        descendants: Set = {cls}
        try:
            for desc in cls.descendants():
                if hasattr(desc, "name"):
                    descendants.add(desc)
        except Exception:
            pass
        return descendants

    def _compute_ic(self, cls) -> float:
        """Information Content: IC(c) = -log2(P(c))

        P(c) = |descendants(c)| / |total_concepts|
        IC가 높을수록 더 구체적(specific)인 개념이다.
        """
        n_desc = self._concept_descendants_count.get(cls, 1)
        p = n_desc / self._total_concepts
        if p <= 0 or p > 1:
            return 0.0
        return -math.log2(p)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LCA (Lowest Common Ancestor)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_lca(self, cls_a, cls_b):
        """두 클래스의 Lowest Common Ancestor를 반환한다.

        LCA = 공통 ancestor 중 depth가 가장 깊은 것.
        """
        key = (id(cls_a), id(cls_b))
        if key in self._lca_cache:
            return self._lca_cache[key]

        ancestors_a = self._get_ancestors(cls_a)
        ancestors_b = self._get_ancestors(cls_b)
        common = ancestors_a & ancestors_b

        if not common:
            self._lca_cache[key] = None
            return None

        lca = max(common, key=lambda c: self._depth_cache.get(c, 0))
        self._lca_cache[key] = lca
        return lca

    def get_depth(self, cls) -> int:
        """클래스의 depth를 반환한다."""
        return self._depth_cache.get(cls, 0)

    def get_ic(self, cls) -> float:
        """클래스의 Information Content를 반환한다."""
        return self._ic_cache.get(cls, 0.0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Wu-Palmer Semantic Similarity
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def wu_palmer_similarity(self, cls_a, cls_b) -> float:
        """Wu-Palmer similarity: 2 * depth(LCA) / (depth(a) + depth(b))

        범위: [0, 1]. 같은 클래스면 1.0.
        """
        if cls_a is cls_b:
            return 1.0

        lca = self.get_lca(cls_a, cls_b)
        if lca is None:
            return 0.0

        d_lca = self.get_depth(lca)
        d_a = self.get_depth(cls_a)
        d_b = self.get_depth(cls_b)

        denom = d_a + d_b
        if denom == 0:
            return 1.0 if d_lca == 0 else 0.0

        return (2.0 * d_lca) / denom

    def lin_similarity(self, cls_a, cls_b) -> float:
        """Lin's similarity: 2 * IC(LCA) / (IC(a) + IC(b))

        IC 기반 시맨틱 유사도. 범위 [0, 1].
        """
        if cls_a is cls_b:
            return 1.0

        lca = self.get_lca(cls_a, cls_b)
        if lca is None:
            return 0.0

        ic_lca = self.get_ic(lca)
        ic_a = self.get_ic(cls_a)
        ic_b = self.get_ic(cls_b)

        denom = ic_a + ic_b
        if denom == 0:
            return 1.0
        return (2.0 * ic_lca) / denom

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Taxonomic Informativeness (TI), ATI
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def taxonomic_informativeness(self, cls) -> float:
        """단일 개념의 Taxonomic Informativeness.

        TI(c) = IC(c) × depth_factor(c)
        depth_factor(c) = depth(c) / max_depth  (정규화)

        깊은 곳에 있고 구체적인 개념일수록 TI가 높다.
        """
        ic = self.get_ic(cls)
        depth = self.get_depth(cls)
        max_depth = max(self._depth_cache.values()) if self._depth_cache else 1
        if max_depth == 0:
            max_depth = 1
        depth_factor = depth / max_depth
        return ic * depth_factor

    def compute_ati_for_refinement(self, refinement) -> float:
        """리파인먼트의 Average Taxonomic Informativeness를 계산한다.

        refinement가 참조하는 온톨로지 클래스들의 평균 TI.
        """
        if self.onto is None:
            return 0.0

        concepts = self._extract_concepts_from_refinement(refinement)
        if not concepts:
            return 0.0

        ti_sum = 0.0
        count = 0
        for concept in concepts:
            if concept in self._ic_cache:
                ti_sum += self.taxonomic_informativeness(concept)
                count += 1

        return ti_sum / count if count > 0 else 0.0

    def _extract_concepts_from_refinement(self, refinement) -> List:
        """OntologyRefinement에서 참조하는 개념들을 추출한다."""
        concepts = []

        if hasattr(refinement, "concept") and refinement.concept is not None:
            concepts.append(refinement.concept)

        if hasattr(refinement, "ref_type"):
            if refinement.ref_type == "conjunction" and hasattr(refinement, "value"):
                # value는 sub-refinement 튜플
                for sub_ref in (refinement.value or []):
                    concepts.extend(self._extract_concepts_from_refinement(sub_ref))

        return concepts

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PIG (Penalized Information Gain)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def penalized_information_gain(
        self,
        ig: float,
        refinement=None,
        ati_override: Optional[float] = None,
    ) -> float:
        """PIG = IG × log(1 + α × ATI)

        Parameters
        ----------
        ig : float
            기본 Information Gain (entropy/gini 기반).
        refinement : OntologyRefinement | None
            ATI 계산 대상 리파인먼트.
        ati_override : float | None
            직접 지정할 ATI 값 (refinement 없이 사용 시).

        Returns
        -------
        float
            Penalized Information Gain.
        """
        if ati_override is not None:
            ati = ati_override
        elif refinement is not None:
            ati = self.compute_ati_for_refinement(refinement)
        else:
            ati = 0.0

        penalty_factor = math.log(1.0 + self.alpha * ati)

        # PIG = IG × PF
        # PF가 0이면 (ATI=0) → IG를 그대로 반환 (log(1+0)=0이므로 곱하면 0)
        # 이를 방지하기 위해 최소 PF=1 보장: max(1.0, PF) 또는 PF+1 사용
        # 논문 원칙: ATI가 높을수록 보상을 주는 방식이므로 1+PF를 곱함
        pig = ig * (1.0 + penalty_factor)
        return pig

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Semantic Similarity-based Splitting Score
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def compute_subset_semantic_similarity(
        self,
        instances: List,
        get_concepts_fn: Callable,
    ) -> float:
        """인스턴스 부분집합의 평균 intra-similarity를 계산한다.

        Sim(a_u) = 모든 (i, j) 쌍의 Wu-Palmer similarity 평균

        Parameters
        ----------
        instances : List
            인스턴스 리스트.
        get_concepts_fn : Callable
            인스턴스 → 관련 온톨로지 클래스 리스트를 반환하는 함수.
            예: lambda inst: [c for c in inst.is_a if hasattr(c, 'name')]

        Returns
        -------
        float
            평균 intra-similarity (0~1).
        """
        if len(instances) <= 1:
            return 1.0

        all_concepts = []
        for inst in instances:
            concepts = get_concepts_fn(inst)
            if concepts:
                all_concepts.append(concepts)

        if len(all_concepts) <= 1:
            return 1.0

        # All pairwise similarities
        total_sim = 0.0
        n_pairs = 0

        for i in range(len(all_concepts)):
            for j in range(i + 1, len(all_concepts)):
                # 두 인스턴스의 개념 집합 간 최대 유사도
                max_sim = 0.0
                for ca in all_concepts[i]:
                    for cb in all_concepts[j]:
                        sim = self.wu_palmer_similarity(ca, cb)
                        if sim > max_sim:
                            max_sim = sim
                total_sim += max_sim
                n_pairs += 1

        return total_sim / n_pairs if n_pairs > 0 else 0.0

    def semantic_similarity_split_score(
        self,
        left_instances: List,
        right_instances: List,
        get_concepts_fn: Callable,
    ) -> float:
        """시맨틱 유사도 기반 분할 점수.

        Sim(A) = Σ_u p_u × Sim(a_u)

        각 부분집합의 intra-similarity를 크기 비율로 가중 합산한다.
        점수가 높을수록 각 부분집합이 온톨로지적으로 동질적이다.

        Parameters
        ----------
        left_instances : List
            왼쪽(만족) 부분집합.
        right_instances : List
            오른쪽(불만족) 부분집합.
        get_concepts_fn : Callable
            인스턴스 → 온톨로지 클래스 리스트 함수.

        Returns
        -------
        float
            가중 평균 시맨틱 유사도 점수.
        """
        total = len(left_instances) + len(right_instances)
        if total == 0:
            return 0.0

        p_left = len(left_instances) / total
        p_right = len(right_instances) / total

        sim_left = self.compute_subset_semantic_similarity(
            left_instances, get_concepts_fn
        )
        sim_right = self.compute_subset_semantic_similarity(
            right_instances, get_concepts_fn
        )

        return p_left * sim_left + p_right * sim_right

    def combined_pig_semantic_score(
        self,
        ig: float,
        left_instances: List,
        right_instances: List,
        refinement=None,
        get_concepts_fn: Optional[Callable] = None,
        semantic_weight: float = 0.3,
    ) -> float:
        """PIG + Semantic Similarity를 결합한 통합 점수.

        Score = (1 - w) × PIG + w × Sim(A) × IG_max_scale

        Parameters
        ----------
        ig : float
            기본 IG.
        left_instances, right_instances : List
            분할 결과.
        refinement : OntologyRefinement | None
            ATI 계산 대상.
        get_concepts_fn : Callable | None
            인스턴스 → 개념 리스트 함수.
        semantic_weight : float
            Semantic Similarity의 가중치 (0~1). 기본 0.3.

        Returns
        -------
        float
            결합 점수.
        """
        pig = self.penalized_information_gain(ig, refinement=refinement)

        if get_concepts_fn is not None:
            sim_score = self.semantic_similarity_split_score(
                left_instances, right_instances, get_concepts_fn
            )
        else:
            sim_score = 0.0

        # IG 스케일에 맞추기 위해 sim_score에 IG를 곱함
        combined = (1.0 - semantic_weight) * pig + semantic_weight * sim_score * max(ig, 0.01)
        return combined


class GraphTaxonomyScorer:
    """NetworkX DiGraph(OWLGraphBuilder 결과) 기반 TI/Similarity 계산기.

    ACO RuleExtractionEngine과 함께 사용한다.
    owlready2 없이 그래프 구조만으로 계층 정보를 추출한다.

    Parameters
    ----------
    graph : nx.DiGraph
        OWLGraphBuilder가 생성한 온톨로지 그래프.
    alpha : float
        PIG에서 ATI의 가중치.
    """

    def __init__(self, graph, alpha: float = 1.0) -> None:
        import networkx as nx

        self.graph = graph
        self.alpha = alpha

        # subClassOf 엣지만 추출하여 계층 그래프 생성
        self._hierarchy = nx.DiGraph()
        for u, v, edata in graph.edges(data=True):
            if edata.get("predicate") == "subClassOf":
                # u → v = child → parent
                self._hierarchy.add_edge(u, v)

        # 모든 노드에 대해 depth 사전 계산
        self._depth_cache: Dict[str, int] = {}
        self._ic_cache: Dict[str, float] = {}
        self._ancestor_cache: Dict[str, Set[str]] = {}

        all_nodes = set(self._hierarchy.nodes())
        # root = depth 0인 노드 (incoming edge가 없는 것)
        self._roots = {n for n in all_nodes if self._hierarchy.in_degree(n) == 0}

        self._total_nodes = max(len(all_nodes), 1)
        # BFS로 depth 계산 (parent → children 방향이 아니라 child → parent 이므로 역방향)
        self._reverse_hierarchy = self._hierarchy.reverse()
        self._compute_all_depths()
        self._compute_all_ic()

        logger.info(
            "GraphTaxonomyScorer: %d hierarchy nodes, %d roots, max_depth=%d",
            len(all_nodes),
            len(self._roots),
            max(self._depth_cache.values()) if self._depth_cache else 0,
        )

    def _compute_all_depths(self) -> None:
        """모든 노드의 depth를 BFS로 계산한다."""
        import networkx as nx

        # root에서 시작하여 children으로 내려가면서 depth 부여
        for root in self._roots:
            self._depth_cache[root] = 0

        for root in self._roots:
            # reverse_hierarchy: parent → children 방향
            try:
                for node in nx.bfs_tree(self._reverse_hierarchy, root):
                    if node not in self._depth_cache:
                        # parent의 depth + 1
                        parents = list(self._hierarchy.successors(node))
                        if parents:
                            self._depth_cache[node] = (
                                max(self._depth_cache.get(p, 0) for p in parents) + 1
                            )
                        else:
                            self._depth_cache[node] = 0
            except Exception:
                continue

        # 연결되지 않은 노드에 기본값
        for node in self._hierarchy.nodes():
            if node not in self._depth_cache:
                self._depth_cache[node] = 0

    def _compute_all_ic(self) -> None:
        """모든 노드의 IC를 계산한다."""
        import networkx as nx

        for node in self._hierarchy.nodes():
            # descendants 수 = reverse_hierarchy에서의 BFS 도달 수
            try:
                desc = nx.descendants(self._reverse_hierarchy, node)
                n_desc = len(desc) + 1  # 자기 자신 포함
            except Exception:
                n_desc = 1

            p = n_desc / self._total_nodes
            if 0 < p <= 1:
                self._ic_cache[node] = -math.log2(p)
            else:
                self._ic_cache[node] = 0.0

    def get_depth(self, node: str) -> int:
        return self._depth_cache.get(node, 0)

    def get_ic(self, node: str) -> float:
        return self._ic_cache.get(node, 0.0)

    def get_ancestors(self, node: str) -> Set[str]:
        """노드의 모든 ancestor를 반환한다 (자기 자신 포함)."""
        if node in self._ancestor_cache:
            return self._ancestor_cache[node]

        import networkx as nx

        ancestors = {node}
        try:
            ancestors.update(nx.ancestors(self._hierarchy, node))
        except Exception:
            pass

        # _hierarchy에서 child→parent 방향이므로
        # ancestors = 현재 노드에서 도달 가능한 모든 노드 (parent 방향)
        try:
            descendants_in_hierarchy = nx.descendants(self._hierarchy, node)
            ancestors.update(descendants_in_hierarchy)
        except Exception:
            pass

        self._ancestor_cache[node] = ancestors
        return ancestors

    def get_lca(self, node_a: str, node_b: str) -> Optional[str]:
        """두 노드의 Lowest Common Ancestor."""
        anc_a = self.get_ancestors(node_a)
        anc_b = self.get_ancestors(node_b)
        common = anc_a & anc_b
        if not common:
            return None
        return max(common, key=lambda n: self._depth_cache.get(n, 0))

    def wu_palmer_similarity(self, node_a: str, node_b: str) -> float:
        """Wu-Palmer similarity between two graph nodes."""
        if node_a == node_b:
            return 1.0

        lca = self.get_lca(node_a, node_b)
        if lca is None:
            return 0.0

        d_lca = self.get_depth(lca)
        d_a = self.get_depth(node_a)
        d_b = self.get_depth(node_b)

        denom = d_a + d_b
        if denom == 0:
            return 1.0 if d_lca == 0 else 0.0

        return (2.0 * d_lca) / denom

    def taxonomic_informativeness(self, node: str) -> float:
        """단일 노드의 TI = IC × (depth / max_depth)."""
        ic = self.get_ic(node)
        depth = self.get_depth(node)
        max_depth = max(self._depth_cache.values()) if self._depth_cache else 1
        if max_depth == 0:
            max_depth = 1
        return ic * (depth / max_depth)

    def compute_ati_for_feature_nodes(self, feature_nodes: List[str]) -> float:
        """피처 노드 리스트의 Average Taxonomic Informativeness."""
        if not feature_nodes:
            return 0.0

        ti_sum = 0.0
        count = 0
        for node in feature_nodes:
            if node in self._ic_cache or node in self._hierarchy:
                ti_sum += self.taxonomic_informativeness(node)
                count += 1

        return ti_sum / count if count > 0 else 0.0

    def penalized_information_gain(
        self,
        ig: float,
        feature_nodes: Optional[List[str]] = None,
        ati_override: Optional[float] = None,
    ) -> float:
        """PIG = IG × (1 + log(1 + α × ATI))

        feature_nodes: 분할에 사용된 feature 노드 ID 리스트.
        """
        if ati_override is not None:
            ati = ati_override
        elif feature_nodes:
            ati = self.compute_ati_for_feature_nodes(feature_nodes)
        else:
            ati = 0.0

        penalty_factor = math.log(1.0 + self.alpha * ati)
        return ig * (1.0 + penalty_factor)

    def compute_nodes_intra_similarity(self, nodes: List[str]) -> float:
        """노드 집합의 평균 pairwise Wu-Palmer similarity."""
        if len(nodes) <= 1:
            return 1.0

        total_sim = 0.0
        n_pairs = 0

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                total_sim += self.wu_palmer_similarity(nodes[i], nodes[j])
                n_pairs += 1

        return total_sim / n_pairs if n_pairs > 0 else 0.0
