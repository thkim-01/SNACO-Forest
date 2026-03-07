"""
ACO Rule Extraction Engine — 개미 경로 탐색 및 의사결정 규칙 추출

개미가 온톨로지 그래프를 탐색하면서 **정보 이득(Information Gain)** 기반
휴리스틱으로 feature 노드를 선택하고, 방문한 feature 노드 시퀀스를
IF-THEN 의사결정 규칙으로 변환한다.

핵심 흐름:
    1. 개미가 그래프 위를 확률적으로 이동 (τ^α · η^β)
    2. is_feature == True 노드 도착 시, 해당 특성의 최적 임계값을 계산
    3. 정보 이득이 min_gain 미만이면 탐색 조기 종료
    4. 방문한 feature 노드 시퀀스 → DecisionPath 규칙 객체로 변환
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 클래스: 단일 분할 조건 + 의사결정 경로
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class SplitCondition:
    """단일 분할 조건: ``Feature <= Threshold`` 또는 ``Feature > Threshold``."""

    feature: str
    operator: Literal["<=", ">"]
    threshold: float
    info_gain: float = 0.0

    def evaluate(self, value: float) -> bool:
        """주어진 값이 이 조건을 만족하는지 판별한다."""
        if self.operator == "<=":
            return value <= self.threshold
        return value > self.threshold

    def __repr__(self) -> str:
        return (
            f"{self.feature} {self.operator} {self.threshold:.4f} "
            f"(IG={self.info_gain:.4f})"
        )


@dataclass
class DecisionPath:
    """개미 한 마리가 생성한 하나의 의사결정 경로.

    Attributes
    ----------
    conditions : list[SplitCondition]
        IF 절을 구성하는 분할 조건의 순서 리스트.
    prediction : int | float | None
        이 규칙이 적용되는 리프에서의 예측값 (다수결 라벨).
    coverage : int
        이 규칙이 커버하는 샘플 수.
    accuracy : float
        이 규칙의 정확도 (커버 샘플 내 정확 예측 비율).
    raw_path : list[str]
        개미가 실제로 방문한 전체 노드 시퀀스 (feature 아닌 것도 포함).
    total_info_gain : float
        경로 상 모든 분할의 정보 이득 합계.
    """

    conditions: List[SplitCondition] = field(default_factory=list)
    prediction: Any = None
    coverage: int = 0
    accuracy: float = 0.0
    raw_path: List[str] = field(default_factory=list)
    total_info_gain: float = 0.0

    # ── Class-imbalance 보정 필드 (Task 2) ──
    leaf_class_dist: Dict[int, int] = field(default_factory=dict)
    """리프 노드의 클래스별 샘플 수. {0: 80, 1: 12}"""
    f1_score: float = 0.0
    """이 규칙의 F1-score (이진 분류 기준, positive class=1)."""
    balanced_accuracy: float = 0.0
    """이 규칙의 Balanced Accuracy."""

    @property
    def depth(self) -> int:
        """규칙의 깊이 (조건 수)."""
        return len(self.conditions)

    @property
    def fitness(self) -> float:
        """경로 품질 지표: F1-score 기반 적합도.

        기존 accuracy 기반에서 F1-score 기반으로 교체.
        전부 0으로 예측(Mode Collapse)하면 Recall=0 → F1=0 → fitness=0
        이므로, 소수 클래스를 올바르게 분류하는 경로만 높은 점수를 받는다.
        """
        if self.coverage == 0:
            return 0.0
        # F1 기반 fitness (F1이 0이면 강력한 패널티)
        # 전부 class 0을 예측하면 F1=0 → fitness ≈ 0.
        # accuracy 기반 fallback은 0.01로 최소화하여
        # F1 > 0인 규칙이 항상 우선 선택되도록 한다.
        base_score = self.f1_score if self.f1_score > 0 else self.accuracy * 0.01
        return base_score * math.log1p(self.coverage)

    @property
    def leaf_probability(self) -> float:
        """리프 노드의 Class 1(양성) 확률.

        이 값이 predict_proba에서 사용된다.
        """
        if not self.leaf_class_dist:
            return 0.5
        total = sum(self.leaf_class_dist.values())
        if total == 0:
            return 0.5
        pos_count = self.leaf_class_dist.get(1, 0)
        return pos_count / total

    def to_rule_string(self) -> str:
        """IF-THEN 규칙 문자열로 변환.

        Examples
        --------
        >>> path.to_rule_string()
        'IF [logp <= 2.3400] AND [tpsa > 75.0000] THEN predict=1'
        """
        if not self.conditions:
            return f"DEFAULT → predict={self.prediction}"
        clauses = [
            f"[{c.feature} {c.operator} {c.threshold:.4f}]"
            for c in self.conditions
        ]
        rule = "IF " + " AND ".join(clauses) + f" THEN predict={self.prediction}"
        return rule

    def evaluate_sample(self, features: Dict[str, float]) -> Optional[Any]:
        """단일 샘플이 이 규칙의 모든 조건을 만족하면 prediction을 반환.

        Parameters
        ----------
        features : dict[str, float]
            특성명 → 수치값.

        Returns
        -------
        prediction | None
            모든 조건 만족 시 prediction, 아니면 None.
        """
        for cond in self.conditions:
            val = features.get(cond.feature)
            if val is None or not cond.evaluate(val):
                return None
        return self.prediction

    def __repr__(self) -> str:
        return (
            f"DecisionPath(depth={self.depth}, coverage={self.coverage}, "
            f"accuracy={self.accuracy:.4f}, "
            f"total_IG={self.total_info_gain:.4f})"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 정보 이론 유틸리티
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """클래스 불균형 보정을 위한 가중치를 계산한다.

    W_c = N_total / (n_classes * N_c)
    소수 클래스(양성)에 높은 가중치, 다수 클래스(음성)에 낮은 가중치.

    Returns
    -------
    dict[int, float]
        클래스 → 가중치.
    """
    if len(labels) == 0:
        return {}
    classes, counts = np.unique(labels, return_counts=True)
    n_total = len(labels)
    n_classes = len(classes)
    weights = {}
    for cls, cnt in zip(classes, counts):
        weights[int(cls)] = n_total / (n_classes * cnt)
    return weights


def _entropy(
    labels: np.ndarray,
    class_weights: Optional[Dict[int, float]] = None,
) -> float:
    """Class-weight 보정이 적용된 Shannon 엔트로피.

    가중치 적용 시: 각 샘플의 가중합으로 확률을 계산하여
    소수 클래스의 불순도 기여를 증폭시킨다.
    """
    if len(labels) == 0:
        return 0.0
    classes, counts = np.unique(labels, return_counts=True)

    if class_weights is not None and len(class_weights) > 0:
        weighted_counts = np.array([
            counts[i] * class_weights.get(int(classes[i]), 1.0)
            for i in range(len(classes))
        ])
        total = weighted_counts.sum()
        if total == 0:
            return 0.0
        probs = weighted_counts / total
    else:
        probs = counts / counts.sum()

    return -float(np.sum(probs * np.log2(probs + 1e-12)))


def _gini(
    labels: np.ndarray,
    class_weights: Optional[Dict[int, float]] = None,
) -> float:
    """Class-weight 보정이 적용된 지니 불순도."""
    if len(labels) == 0:
        return 0.0
    classes, counts = np.unique(labels, return_counts=True)

    if class_weights is not None and len(class_weights) > 0:
        weighted_counts = np.array([
            counts[i] * class_weights.get(int(classes[i]), 1.0)
            for i in range(len(classes))
        ])
        total = weighted_counts.sum()
        if total == 0:
            return 0.0
        probs = weighted_counts / total
    else:
        probs = counts / counts.sum()

    return float(1.0 - np.sum(probs ** 2))


def _gain_ratio(
    parent_labels: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    class_weights: Optional[Dict[int, float]] = None,
) -> float:
    """Gain Ratio (C5.0 기준).

    GR = IG_entropy / SplitInfo
    SplitInfo = -Σ (|S_i|/|S|) log2(|S_i|/|S|)
    """
    ig = _info_gain(parent_labels, left_labels, right_labels, "entropy", class_weights)
    n = len(parent_labels)
    if n == 0:
        return 0.0
    n_left = len(left_labels)
    n_right = len(right_labels)
    ratios = []
    for ni in (n_left, n_right):
        if ni > 0:
            r = ni / n
            ratios.append(r)
    split_info = -sum(r * np.log2(r + 1e-12) for r in ratios)
    if split_info < 1e-12:
        return ig
    return ig / split_info


def _chi_square(
    parent_labels: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    class_weights: Optional[Dict[int, float]] = None,
) -> float:
    """Chi-square statistic (CHAID 기준).

    카이제곱 통계량으로 분할 품질을 측정한다.
    값이 클수록 분할이 유의미하다.
    """
    n = len(parent_labels)
    if n == 0:
        return 0.0
    classes = np.unique(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0

    chi2 = 0.0
    for cls in classes:
        # 전체에서 이 클래스의 비율
        p_cls = np.sum(parent_labels == cls) / n
        for subset, n_sub in ((left_labels, n_left), (right_labels, n_right)):
            expected = p_cls * n_sub
            observed = np.sum(subset == cls)
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected

    # IG와 스케일 맞추기: 정규화 (0~1 범위)
    # 자유도 = (classes - 1) * (splits - 1) = (K-1)*1
    dof = max(len(classes) - 1, 1)
    return chi2 / (n * dof + 1e-12)


def _info_gain(
    parent_labels: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    criterion: str = "entropy",
    class_weights: Optional[Dict[int, float]] = None,
    *,
    graph_scorer=None,
    feature_node: Optional[str] = None,
    feature_nodes: Optional[List[str]] = None,
    pig_alpha: Optional[float] = None,
    semantic_weight: float = 0.3,
) -> float:
    """Class-weight 보정이 적용된 정보 이득.

    IG = H_w(parent) - [W_L/W_total · H_w(L) + W_R/W_total · H_w(R)]

    가중치 적용 시 분할 비율도 가중 샘플 수로 계산한다.

    criterion 확장:
        - ``"entropy"`` — Shannon 엔트로피 기반 IG (ID3)
        - ``"gini"`` — 지니 불순도 기반 IG (CART)
        - ``"gain_ratio"`` — Gain Ratio (C5.0)
        - ``"chi_square"`` — 카이제곱 통계량 (CHAID)
        - ``"pig"`` — Penalized Information Gain (계층 인식 분할)
        - ``"semantic_similarity"`` — 시맨틱 유사도 기반 분할
        - ``"pig_semantic"`` — PIG + Semantic Similarity 결합
    """
    # C5.0 / CHAID는 별도 함수로 분기
    if criterion == "gain_ratio":
        return _gain_ratio(parent_labels, left_labels, right_labels, class_weights)
    if criterion == "chi_square":
        return _chi_square(parent_labels, left_labels, right_labels, class_weights)

    # PIG / Semantic Similarity / PIG+Semantic
    if criterion in ("pig", "semantic_similarity", "pig_semantic"):
        from src.aco.taxonomy_aware_gain import (
            pig_gain,
            semantic_sim_gain,
            pig_semantic_combined_gain,
        )
        _nodes = list(feature_nodes or [])
        if feature_node and feature_node not in _nodes:
            _nodes.append(feature_node)
        if criterion == "pig":
            return pig_gain(
                parent_labels, left_labels, right_labels,
                feature_nodes=_nodes or None,
                graph_scorer=graph_scorer,
                base_criterion="entropy",
                class_weights=class_weights,
                alpha=pig_alpha,
            )
        elif criterion == "semantic_similarity":
            return semantic_sim_gain(
                parent_labels, left_labels, right_labels,
                left_feature_nodes=_nodes or None,
                right_feature_nodes=_nodes or None,
                graph_scorer=graph_scorer,
                base_criterion="entropy",
                class_weights=class_weights,
                semantic_weight=semantic_weight,
            )
        else:  # pig_semantic
            return pig_semantic_combined_gain(
                parent_labels, left_labels, right_labels,
                feature_nodes=_nodes or None,
                left_feature_nodes=_nodes or None,
                right_feature_nodes=_nodes or None,
                graph_scorer=graph_scorer,
                base_criterion="entropy",
                class_weights=class_weights,
                alpha=pig_alpha,
                semantic_weight=semantic_weight,
            )

    impurity_fn = _entropy if criterion == "entropy" else _gini
    n = len(parent_labels)
    if n == 0:
        return 0.0

    h_parent = impurity_fn(parent_labels, class_weights)

    n_left, n_right = len(left_labels), len(right_labels)

    if class_weights is not None and len(class_weights) > 0:
        # 가중 샘플 수로 분할 비율 계산
        w_left = sum(class_weights.get(int(l), 1.0) for l in left_labels)
        w_right = sum(class_weights.get(int(l), 1.0) for l in right_labels)
        w_total = w_left + w_right
        if w_total == 0:
            return 0.0
        ratio_left = w_left / w_total
        ratio_right = w_right / w_total
    else:
        ratio_left = n_left / n
        ratio_right = n_right / n

    h_left = impurity_fn(left_labels, class_weights) if n_left > 0 else 0.0
    h_right = impurity_fn(right_labels, class_weights) if n_right > 0 else 0.0

    return h_parent - (ratio_left * h_left + ratio_right * h_right)


def find_best_threshold(
    feature_values: np.ndarray,
    labels: np.ndarray,
    criterion: str = "entropy",
    n_candidates: int = 32,
    class_weights: Optional[Dict[int, float]] = None,
    *,
    graph_scorer=None,
    feature_node: Optional[str] = None,
    pig_alpha: Optional[float] = None,
    semantic_weight: float = 0.3,
) -> Tuple[float, float, str]:
    """Class-weight 보정이 적용된 최적 분할 임계값 탐색.

    Parameters
    ----------
    feature_values : array-like
        특성의 수치 벡터.
    labels : array-like
        대응하는 라벨 벡터.
    criterion : str
        ``"entropy"`` | ``"gini"`` | ``"gain_ratio"`` | ``"chi_square"``
        | ``"pig"`` | ``"semantic_similarity"`` | ``"pig_semantic"``.
    n_candidates : int
        후보 임계값 수 (분위수 기반 샘플링).
    class_weights : dict[int, float] | None
        클래스별 가중치. None이면 균등 가중치.
    graph_scorer : GraphTaxonomyScorer | None
        PIG/Semantic Similarity 계산기. pig/semantic 기준 사용 시 필요.
    feature_node : str | None
        분할에 사용된 feature 노드 ID (PIG ATI 계산용).
    pig_alpha : float | None
        PIG α 하이퍼파라미터.
    semantic_weight : float
        Semantic Similarity 가중치.

    Returns
    -------
    (threshold, info_gain, operator)
    """
    valid_mask = ~(np.isnan(feature_values) | np.isnan(labels.astype(float)))
    fv = feature_values[valid_mask]
    lb = labels[valid_mask]

    if len(fv) < 2:
        return float(np.nanmedian(feature_values)), 0.0, "<="

    # 분위수 기반 후보 임계값
    percentiles = np.linspace(5, 95, n_candidates)
    thresholds = np.unique(np.percentile(fv, percentiles))

    best_t, best_ig, best_op = float(np.median(fv)), 0.0, "<="

    for t in thresholds:
        left_mask = fv <= t
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue
        ig = _info_gain(
            lb, lb[left_mask], lb[right_mask], criterion, class_weights,
            graph_scorer=graph_scorer,
            feature_node=feature_node,
            pig_alpha=pig_alpha,
            semantic_weight=semantic_weight,
        )
        if ig > best_ig:
            best_ig = ig
            best_t = float(t)
            best_op = "<="

    return best_t, best_ig, best_op


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RuleExtractionEngine 핵심 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RuleExtractionEngine:
    """개미 경로 탐색 + 의사결정 규칙 추출 엔진.

    Parameters
    ----------
    graph : nx.DiGraph
        OWLGraphBuilder로 빌드되고 SMILESOntologyBridge로
        feature 노드가 태그된 온톨로지 그래프.
    feature_df : pd.DataFrame
        SMILES 특성 데이터프레임. 각 컬럼은 특성명, 값은 수치.
    label_series : pd.Series
        각 샘플의 라벨 (분류 타겟). feature_df와 인덱스 정렬.
    alpha : float
        페로몬 지수, 기본 1.0.
    beta : float
        휴리스틱 지수, 기본 2.0.
    max_path_length : int
        개미가 수집할 최대 feature 노드 수 = 규칙 최대 깊이. 기본 5.
    max_steps : int
        그래프 위 최대 이동 횟수 (feature 아닌 노드 포함). 기본 100.
    min_gain : float
        정보 이득 하한 — 이보다 작으면 탐색 조기 종료. 기본 0.01.
    min_samples_leaf : int
        분할 후 리프에 남아야 하는 최소 샘플 수. 기본 5.
    criterion : str
        분할 기준. ``"entropy"`` | ``"gini"`` | ``"pig"`` | ``"semantic_similarity"`` | ``"pig_semantic"``.
        기본 ``"entropy"``.
    jump_penalty_base : float
        계층 건너뛰기 감가율 기본값 (0 <= p <= 1).
        예: 0.9면 건너뛰기 시 η에 0.9 배율 적용.
    jump_gamma : float
        계층 건너뛰기 페널티 강도 γ.
        γ=0 이면 페널티 없음, γ가 커질수록 건너뛰기 억제,
        γ=∞ 이면 건너뛰기 확률이 0에 수렴.
    pig_alpha : float
        PIG의 ATI 가중치 α. 기본 1.0.
    semantic_weight : float
        Semantic Similarity 가중치 (0~1). 기본 0.3.
    seed : int | None
        재현성 시드.

    Examples
    --------
    >>> engine = RuleExtractionEngine(G, feature_df, labels)
    >>> path = engine.extract_single_rule()
    >>> print(path.to_rule_string())
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        feature_df: pd.DataFrame,
        label_series: pd.Series,
        *,
        alpha: float = 1.0,
        beta: float = 2.0,
        max_path_length: int = 5,
        max_steps: int = 100,
        min_gain: float = 0.01,
        min_samples_leaf: int = 5,
        criterion: str = "entropy",
        jump_penalty_base: float = 0.9,
        jump_gamma: float = 1.0,
        pig_alpha: float = 1.0,
        semantic_weight: float = 0.3,
        seed: Optional[int] = None,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> None:
        self.graph = graph
        self.feature_df = feature_df.copy()
        self.labels = label_series.values.copy()
        self.feature_columns = list(feature_df.columns)

        self.alpha = alpha
        self.beta = beta
        self.max_path_length = max_path_length
        self.max_steps = max_steps
        self.min_gain = min_gain
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.pig_alpha = float(pig_alpha)
        self.semantic_weight = float(semantic_weight)
        # GraphTaxonomyScorer: PIG/SemanticSim 사용 시 계층 정보 제공
        self._graph_scorer = None
        if criterion in ("pig", "semantic_similarity", "pig_semantic"):
            try:
                from src.sdt.taxonomy_scorer import GraphTaxonomyScorer
                self._graph_scorer = GraphTaxonomyScorer(
                    graph, alpha=self.pig_alpha
                )
            except Exception as e:
                logger.warning("GraphTaxonomyScorer init failed: %s. Falling back to entropy.", e)
        if not (0.0 <= jump_penalty_base <= 1.0):
            raise ValueError(
                f"jump_penalty_base must be in [0, 1], got {jump_penalty_base}"
            )
        if jump_gamma < 0:
            raise ValueError(f"jump_gamma must be >= 0, got {jump_gamma}")
        self.jump_penalty_base = float(jump_penalty_base)
        self.jump_gamma = float(jump_gamma)

        # ── Class Weight: 불균형 보정 (자동 계산 또는 외부 지정) ──
        if class_weights is not None:
            self.class_weights = class_weights
        else:
            # 자동 계산: W_1 = N_neg / N_pos
            self.class_weights = compute_class_weights(self.labels)

        self._rng = random.Random(seed)
        self._np_rng = np.random.RandomState(seed)

        # child -> parent (subClassOf) 전용 계층 그래프
        self._hierarchy_graph = nx.DiGraph()
        for u, v, edata in self.graph.edges(data=True):
            if edata.get("predicate") == "subClassOf":
                self._hierarchy_graph.add_edge(u, v)
        self._jump_distance_cache: Dict[Tuple[str, str], int] = {}

        # feature_key → DataFrame 컬럼명 역매핑
        self._feat_to_col: Dict[str, str] = {}
        for nid, ndata in graph.nodes(data=True):
            if ndata.get("is_feature"):
                fkey = ndata.get("feature_key") or ndata.get("feature_name", "")
                # DataFrame에 해당 컬럼이 존재하는지 확인
                if fkey in self.feature_df.columns:
                    self._feat_to_col[nid] = fkey
                else:
                    # 부분 매칭 시도
                    for col in self.feature_df.columns:
                        if col.lower() == fkey.lower():
                            self._feat_to_col[nid] = col
                            break

        # feature 노드 간 완전 연결 서브그래프 구축
        # → 개미가 feature 노드 사이를 직접 이동할 수 있게 함
        self._build_feature_subgraph()

        logger.info(
            "RuleExtractionEngine initialized: %d feature nodes mapped to df cols, "
            "%d samples, %d features, criterion=%s, jump_penalty=%.3f, gamma=%.3f",
            len(self._feat_to_col),
            len(self.labels),
            len(self.feature_columns),
            self.criterion,
            self.jump_penalty_base,
            self.jump_gamma,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 공개 API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def extract_single_rule(
        self,
        start_node: Optional[str] = None,
        *,
        explore_mode: str = "auto",
    ) -> DecisionPath:
        """개미 한 마리를 탐색시키고, 의사결정 경로(규칙)를 반환한다.

        Parameters
        ----------
        start_node : str | None
            출발 노드. None이면 explore_mode에 따라 결정.
        explore_mode : str
            ``"auto"`` — feature-direct 70%, hub 20%, random 10%.
            ``"hub"`` — hub(Thing) 노드에서 시작.
            ``"feature"`` — 임의의 feature 노드에서 시작.
            ``"random"`` — 완전 랜덤.

        Returns
        -------
        DecisionPath
        """
        G = self.graph

        # 출발 노드 결정
        if start_node is None:
            start_node = self._pick_start_node(explore_mode)

        # ── 그래프 탐색 ──
        raw_path: List[str] = [start_node]
        visited: Set[str] = {start_node}
        current = start_node

        # feature 노드 방문 기록
        feature_visits: List[str] = []

        # 현재 적용 중인 샘플 마스크 (규칙에 의해 점점 좁혀짐)
        active_mask = np.ones(len(self.labels), dtype=bool)

        conditions: List[SplitCondition] = []

        for step in range(self.max_steps):
            # ── 최대 깊이 도달 확인 ──
            if len(feature_visits) >= self.max_path_length:
                logger.debug("Max path length reached (%d).", self.max_path_length)
                break

            # ── 남은 샘플이 min_samples_leaf 미만이면 종료 ──
            if active_mask.sum() < self.min_samples_leaf * 2:
                logger.debug(
                    "Too few samples remaining (%d) — stopping.",
                    active_mask.sum(),
                )
                break

            # ── 후보 노드 수집 ──
            candidates = self._get_candidates(current, visited)
            if not candidates:
                break

            # ── 다음 노드 선택 (ACO 확률 규칙) ──
            next_node = self._select_next(current, candidates, active_mask)
            raw_path.append(next_node)
            visited.add(next_node)
            current = next_node

            # ── feature 노드 도착 시 분할 수행 ──
            if current in self._feat_to_col:
                col_name = self._feat_to_col[current]
                fv = self.feature_df[col_name].values
                lb = self.labels

                # 현재 활성 샘플 내에서 최적 임계값 탐색
                active_fv = fv[active_mask]
                active_lb = lb[active_mask]

                threshold, ig, operator = find_best_threshold(
                    active_fv, active_lb, self.criterion,
                    class_weights=self.class_weights,
                    graph_scorer=self._graph_scorer,
                    feature_node=current,
                    pig_alpha=self.pig_alpha,
                    semantic_weight=self.semantic_weight,
                )

                # ── 정보 이득 하한 체크 (조기 종료) ──
                if ig < self.min_gain:
                    logger.debug(
                        "Feature '%s': IG=%.4f < min_gain=%.4f — skipping.",
                        col_name,
                        ig,
                        self.min_gain,
                    )
                    continue

                # 분할 방향 결정: <= / > 중 더 정보 이득이 높은 쪽
                split_mask = fv <= threshold

                # 활성 마스크에 AND 결합
                new_left = active_mask & split_mask
                new_right = active_mask & (~split_mask)

                # 분기 방향 결정: class_weights가 있으면 가중 순도 기반
                # 확률적 선택으로 다양한 경로 탐색
                if self.class_weights and new_left.sum() > 0 and new_right.sum() > 0:
                    left_wp = _weighted_purity(lb[new_left], self.class_weights)
                    right_wp = _weighted_purity(lb[new_right], self.class_weights)
                    total_wp = left_wp + right_wp
                    if total_wp > 0:
                        # 가중 순도에 비례한 확률적 분기
                        go_left = random.random() < (left_wp / total_wp)
                    else:
                        go_left = True
                else:
                    left_purity = _purity(lb[new_left]) if new_left.sum() > 0 else 0
                    right_purity = _purity(lb[new_right]) if new_right.sum() > 0 else 0
                    go_left = left_purity >= right_purity

                if go_left:
                    chosen_op = "<="
                    active_mask = new_left
                else:
                    chosen_op = ">"
                    active_mask = new_right

                cond = SplitCondition(
                    feature=col_name,
                    operator=chosen_op,
                    threshold=threshold,
                    info_gain=ig,
                )
                conditions.append(cond)
                feature_visits.append(current)

                logger.debug(
                    "Split: %s %s %.4f (IG=%.4f, remaining=%d)",
                    col_name,
                    chosen_op,
                    threshold,
                    ig,
                    active_mask.sum(),
                )

        # ── DecisionPath 조립 ──
        active_labels = self.labels[active_mask]
        if len(active_labels) > 0:
            # Class weight 반영 가중 다수결: 소수 클래스가 일정 비율
            # 이상이면 소수 클래스를 예측하여 Mode Collapse 방지
            prediction = int(_weighted_majority_class(
                active_labels, self.class_weights,
            ))
            correct = (active_labels == prediction).sum()
            accuracy = float(correct / len(active_labels))
        else:
            prediction = int(_weighted_majority_class(
                self.labels, self.class_weights,
            ))
            accuracy = 0.0

        total_ig = sum(c.info_gain for c in conditions)

        # ── 리프 노드 클래스 분포 및 F1/BalancedAcc 계산 (Task 2) ──
        leaf_class_dist: Dict[int, int] = {}
        f1 = 0.0
        bal_acc = 0.0
        if len(active_labels) > 0:
            classes, counts = np.unique(active_labels, return_counts=True)
            leaf_class_dist = {int(c): int(n) for c, n in zip(classes, counts)}

            # 전체 학습 데이터에 대해 이 규칙의 예측을 평가
            all_preds = np.full(len(self.labels), -1)
            for idx in range(len(self.labels)):
                match = True
                for cond in conditions:
                    val = self.feature_df[cond.feature].iloc[idx]
                    if not cond.evaluate(val):
                        match = False
                        break
                if match:
                    all_preds[idx] = prediction

            # F1-score 계산 (커버된 샘플만, fn_uncovered 제외)
            # 개별 규칙의 F1은 커버 범위 내에서만 평가한다.
            # fn_uncovered를 포함하면 좁은 class-1 규칙의 F1이
            # 사실상 0이 되어 다시 Mode Collapse에 빠진다.
            # 숲 전체의 커버리지는 앙상블 단계에서 보장된다.
            covered_mask = all_preds != -1
            if covered_mask.sum() > 0:
                y_t = self.labels[covered_mask]
                y_p = all_preds[covered_mask]
                tp = int(np.sum((y_t == 1) & (y_p == 1)))
                fp = int(np.sum((y_t == 0) & (y_p == 1)))
                fn = int(np.sum((y_t == 1) & (y_p == 0)))
                tn = int(np.sum((y_t == 0) & (y_p == 0)))

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                bal_acc = (tpr + tnr) / 2.0

        path_obj = DecisionPath(
            conditions=conditions,
            prediction=prediction,
            coverage=int(active_mask.sum()),
            accuracy=accuracy,
            raw_path=raw_path,
            total_info_gain=total_ig,
            leaf_class_dist=leaf_class_dist,
            f1_score=f1,
            balanced_accuracy=bal_acc,
        )

        return path_obj

    def extract_rules(
        self,
        n_ants: int = 20,
        start_node: Optional[str] = None,
        *,
        deduplicate: bool = True,
    ) -> List[DecisionPath]:
        """여러 개미를 탐색시키고, 의사결정 규칙 리스트를 반환한다.

        Parameters
        ----------
        n_ants : int
            탐색할 개미 수.
        start_node : str | None
            공통 출발 노드.
        deduplicate : bool
            동일한 규칙(조건 동일)을 제거할지 여부.

        Returns
        -------
        list[DecisionPath]
            적합도 내림차순 정렬.
        """
        paths: List[DecisionPath] = []
        for i in range(n_ants):
            path = self.extract_single_rule(start_node, explore_mode="auto")
            if path.depth > 0:
                paths.append(path)

                # 좋은 경로의 페로몬 강화
                self._update_pheromones(path)

            logger.debug("Ant %d/%d: %s", i + 1, n_ants, path)

        if deduplicate:
            paths = self._deduplicate(paths)

        # 적합도 내림차순
        paths.sort(key=lambda p: p.fitness, reverse=True)
        return paths

    def evaluate_rules(
        self,
        rules: List[DecisionPath],
        feature_df: Optional[pd.DataFrame] = None,
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """규칙 집합의 전체 성능을 평가한다.

        Parameters
        ----------
        rules : list[DecisionPath]
            평가할 규칙 리스트.
        feature_df : pd.DataFrame, optional
            테스트 데이터. None이면 학습 데이터 사용.
        labels : np.ndarray, optional
            테스트 라벨.

        Returns
        -------
        dict
            ``{n_rules, coverage_rate, avg_accuracy, predictions}``
        """
        if feature_df is None:
            feature_df = self.feature_df
        if labels is None:
            labels = self.labels

        n_samples = len(labels)
        predictions = np.full(n_samples, fill_value=-1)
        covered = np.zeros(n_samples, dtype=bool)

        for rule in rules:
            for idx in range(n_samples):
                if covered[idx]:
                    continue
                row = {
                    col: feature_df.iloc[idx][col]
                    for col in feature_df.columns
                }
                pred = rule.evaluate_sample(row)
                if pred is not None:
                    predictions[idx] = pred
                    covered[idx] = True

        # 미커버 샘플은 다수결
        default_pred = int(_majority_class(labels))
        predictions[~covered] = default_pred

        correct = (predictions == labels).sum()
        accuracy = float(correct / n_samples) if n_samples > 0 else 0.0

        return {
            "n_rules": len(rules),
            "coverage_rate": float(covered.sum() / n_samples) if n_samples > 0 else 0.0,
            "avg_accuracy": accuracy,
            "predictions": predictions,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 내부 구현
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _pick_start_node(self, explore_mode: str) -> str:
        """탐색 모드에 따라 출발 노드를 결정한다."""
        G = self.graph
        feature_nodes = list(self._feat_to_col.keys())

        if explore_mode == "feature" or (
            explore_mode == "auto" and self._rng.random() < 0.70
        ):
            # feature 노드에서 직접 출발 (다양한 규칙 보장)
            if feature_nodes:
                return self._rng.choice(feature_nodes)

        if explore_mode == "hub" or (
            explore_mode == "auto" and self._rng.random() < 0.67
        ):
            # hub 노드에서 출발
            for hc in ["Thing", "thing"]:
                if hc in G:
                    return hc

        # random
        return self._rng.choice(list(G.nodes))

    def _build_feature_subgraph(self) -> None:
        """feature 노드 간 완전 연결(complete) 서브그래프를 구축한다.

        개미가 feature A에서 feature B로 직접 이동할 수 있도록
        모든 feature 노드 쌍 사이에 양방향 엣지를 추가한다.
        이 엣지들은 ``predicate='featureLink'``로 태그된다.
        """
        G = self.graph
        feat_nodes = list(self._feat_to_col.keys())
        added = 0
        for i, u in enumerate(feat_nodes):
            for v in feat_nodes[i + 1:]:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, predicate="featureLink", pheromone=1.0)
                    added += 1
                if not G.has_edge(v, u):
                    G.add_edge(v, u, predicate="featureLink", pheromone=1.0)
                    added += 1
        if added > 0:
            logger.info(
                "Feature subgraph: added %d inter-feature edges among %d feature nodes.",
                added,
                len(feat_nodes),
            )

    def _get_candidates(
        self,
        current: str,
        visited: Set[str],
    ) -> List[str]:
        """현재 노드에서 미방문 이웃 노드들을 반환."""
        G = self.graph
        neighbors = set(G.successors(current)) | set(G.predecessors(current))
        return list(neighbors - visited)

    def _select_next(
        self,
        current: str,
        candidates: List[str],
        active_mask: np.ndarray,
    ) -> str:
        """ACO 확률 전이 규칙.

        P(i→j) ∝ τ_ij^α · η_j^β

        η_j 계산:
        - feature 노드: 해당 특성으로 active 데이터를 분할했을 때의
          정보 이득 (동적 휴리스틱)
        - 일반 노드: 1.0 (기본)
        """
        G = self.graph
        weights: List[float] = []

        for cand in candidates:
            # ── 페로몬 τ ──
            if G.has_edge(current, cand):
                tau = G.edges[current, cand].get("pheromone", 1.0)
            elif G.has_edge(cand, current):
                tau = G.edges[cand, current].get("pheromone", 1.0)
            else:
                tau = 1.0

            # ── 휴리스틱 η + 계층 건너뛰기 페널티 ──
            eta_base = self._compute_heuristic(cand, active_mask)
            jump_penalty = self._compute_jump_penalty(current, cand)
            eta = eta_base * jump_penalty

            w = (tau ** self.alpha) * (eta ** self.beta)
            weights.append(max(w, 1e-12))

        total = sum(weights)
        if total == 0:
            return self._rng.choice(candidates)

        probs = [w / total for w in weights]
        return self._rng.choices(candidates, weights=probs, k=1)[0]

    def _compute_heuristic(
        self,
        node_id: str,
        active_mask: np.ndarray,
    ) -> float:
        """노드의 휴리스틱 η를 계산한다.

        feature 노드: Class-weight 보정 정보 이득 기반 (높을수록 좋음).
        일반 노드: 기본값 1.0, feature 근처일수록 약간 보너스.

        Class Weight가 적용되어 소수 클래스(독성 물질, Class 1) 분할 시
        다수 클래스보다 훨씬 높은 η를 얻는다.
        """
        DEFAULT_ETA = 1.0
        FEATURE_BASE = 2.0  # feature 노드 최소 η

        if node_id not in self._feat_to_col:
            # 일반 노드: feature 이웃이 있으면 약간의 보너스
            G = self.graph
            nbrs = set(G.successors(node_id)) | set(G.predecessors(node_id))
            feature_neighbor_count = sum(
                1 for n in nbrs if n in self._feat_to_col
            )
            return DEFAULT_ETA + 0.5 * feature_neighbor_count

        # feature 노드 → Class-weight 보정 정보 이득 기반 η
        col_name = self._feat_to_col[node_id]
        fv = self.feature_df[col_name].values[active_mask]
        lb = self.labels[active_mask]

        if len(lb) < 2:
            return FEATURE_BASE

        _, ig, _ = find_best_threshold(
            fv, lb, self.criterion, n_candidates=16,
            class_weights=self.class_weights,
            graph_scorer=self._graph_scorer,
            feature_node=node_id,
            pig_alpha=self.pig_alpha,
            semantic_weight=self.semantic_weight,
        )

        # η = base + IG 스케일링 (IG는 보통 0~1)
        return FEATURE_BASE + ig * 10.0

    def _compute_jump_penalty(self, current: str, cand: str) -> float:
        """부모 노드를 건너뛰고 하위로 직행할 때 η 감가율을 계산한다.

        Jump 정의:
            cand 가 current 의 자손이고(subClassOf 기준),
            current -> cand 직접 subClassOf/hasSubClass 엣지가 없는데
            계층 거리(자손→조상)가 2 이상인 경우.

        수식:
            penalty = p^(γ * skip_levels)
            p = jump_penalty_base (예: 0.9)
            γ = jump_gamma
            skip_levels = distance(descendant, ancestor) - 1
        """
        if self.jump_gamma == 0.0:
            return 1.0

        G = self.graph
        forward_pred = G.edges[current, cand].get("predicate") if G.has_edge(current, cand) else None
        reverse_pred = G.edges[cand, current].get("predicate") if G.has_edge(cand, current) else None

        # 직접 parent-child 이동이면 페널티 없음
        if forward_pred in {"hasSubClass", "subClassOf"} or reverse_pred in {"hasSubClass", "subClassOf"}:
            return 1.0

        key = (cand, current)
        if key in self._jump_distance_cache:
            dist = self._jump_distance_cache[key]
        else:
            try:
                # subClassOf는 child -> parent 이므로 cand(하위) -> current(상위)
                dist = nx.shortest_path_length(self._hierarchy_graph, source=cand, target=current)
            except Exception:
                dist = -1
            self._jump_distance_cache[key] = dist

        # 자손 관계가 아니면(또는 도달 불가) 페널티 없음
        if dist < 2:
            return 1.0

        skip_levels = dist - 1
        if math.isinf(self.jump_gamma):
            return 0.0

        penalty = self.jump_penalty_base ** (self.jump_gamma * skip_levels)
        return max(float(penalty), 1e-12)

    def _update_pheromones(
        self,
        path: DecisionPath,
        deposit_scale: float = 1.0,
        evaporation: float = 0.02,
    ) -> None:
        """경로의 품질(fitness)에 비례하여 페로몬을 갱신.

        - 전역 증발은 extract_rules 전체에서 공유되므로 여기서는 경로 강화만.
        - deposit = fitness × deposit_scale
        """
        G = self.graph
        deposit = path.fitness * deposit_scale

        # 전역 증발 (가벼운 비율)
        for u, v, edata in G.edges(data=True):
            edata["pheromone"] = max(
                edata.get("pheromone", 1.0) * (1 - evaporation), 1e-6
            )

        # 경로 강화
        raw = path.raw_path
        for i in range(len(raw) - 1):
            u, v = raw[i], raw[i + 1]
            if G.has_edge(u, v):
                G.edges[u, v]["pheromone"] += deposit
            if G.has_edge(v, u):
                G.edges[v, u]["pheromone"] += deposit

    @staticmethod
    def _deduplicate(paths: List[DecisionPath]) -> List[DecisionPath]:
        """동일 조건 조합의 중복 경로를 제거한다."""
        seen: Set[str] = set()
        unique: List[DecisionPath] = []
        for p in paths:
            key = "|".join(
                f"{c.feature}{c.operator}{c.threshold:.4f}"
                for c in p.conditions
            )
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸리티 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _majority_class(labels: np.ndarray) -> Any:
    """다수결 클래스 반환."""
    if len(labels) == 0:
        return 0
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]


def _weighted_majority_class(
    labels: np.ndarray,
    class_weights: Optional[Dict[int, float]] = None,
) -> Any:
    """Class weight를 반영한 가중 다수결 클래스.

    가중 카운트 = 실제 카운트 × W_c.
    소수 클래스(양성)의 가중치가 높으므로, 소수 클래스가 일정 비율
    이상이면 소수 클래스가 다수결 승리한다.

    예: W_1=5.0, 80% class 0 / 20% class 1
        → weighted_0 = 80 × 0.556 = 44.5
        → weighted_1 = 20 × 5.0  = 100.0
        → 예측: class 1
    """
    if len(labels) == 0:
        return 0
    if class_weights is None:
        return _majority_class(labels)
    classes, counts = np.unique(labels, return_counts=True)
    weighted = {
        int(c): int(n) * class_weights.get(int(c), 1.0)
        for c, n in zip(classes, counts)
    }
    return max(weighted, key=weighted.get)


def _purity(labels: np.ndarray) -> float:
    """다수결 클래스 비율 (순도)."""
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(np.max(counts) / len(labels))


def _weighted_purity(
    labels: np.ndarray,
    class_weights: Optional[Dict[int, float]] = None,
) -> float:
    """Class weight를 반영한 가중 순도.

    가중 순도 = max(weighted_count_c) / sum(weighted_count_c).
    소수 클래스 weight가 높으면, 소수 클래스가 일정 비율 이상인
    브랜치의 가중 순도가 높아진다.
    """
    if len(labels) == 0:
        return 0.0
    if class_weights is None:
        return _purity(labels)
    classes, counts = np.unique(labels, return_counts=True)
    weighted = [
        int(n) * class_weights.get(int(c), 1.0)
        for c, n in zip(classes, counts)
    ]
    total = sum(weighted)
    if total == 0:
        return 0.0
    return max(weighted) / total
