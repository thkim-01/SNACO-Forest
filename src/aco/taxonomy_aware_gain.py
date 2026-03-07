"""
ACO Pipeline용 Taxonomy-Aware Gain 함수

rule_extraction.py의 _info_gain()과 동일한 인터페이스로,
PIG (Penalized Information Gain) 및 Semantic Similarity 기반
분할 기준을 추가한다.

사용법
------
    from src.aco.taxonomy_aware_gain import pig_gain, semantic_sim_gain

    # PIG: 기본 IG에 온톨로지 계층 보상을 곱함
    gain = pig_gain(parent_labels, left_labels, right_labels,
                    feature_node="hasAromatic",
                    graph_scorer=scorer,
                    base_criterion="entropy")

    # Semantic Similarity: 분할 후 부분집합의 계층적 동질성 측정
    gain = semantic_sim_gain(parent_labels, left_labels, right_labels,
                             left_nodes=["Amine", "Alcohol"],
                             right_nodes=["Aromatic", "Ether"],
                             graph_scorer=scorer,
                             base_criterion="entropy")
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _info_gain_base(
    parent_labels: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    criterion: str = "entropy",
    class_weights: Optional[Dict[int, float]] = None,
) -> float:
    """기본 IG 계산 (rule_extraction._info_gain 과 동일한 로직).

    이 모듈이 rule_extraction.py에 의존하지 않도록 로컬에 복제한다.
    """
    from src.aco.rule_extraction import _info_gain
    return _info_gain(parent_labels, left_labels, right_labels, criterion, class_weights)


def pig_gain(
    parent_labels: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    feature_node: Optional[str] = None,
    feature_nodes: Optional[List[str]] = None,
    graph_scorer=None,
    base_criterion: str = "entropy",
    class_weights: Optional[Dict[int, float]] = None,
    alpha: Optional[float] = None,
) -> float:
    """Penalized Information Gain (PIG).

    PIG(S) = IG(S) × (1 + log(1 + α × ATI))

    Parameters
    ----------
    parent_labels, left_labels, right_labels : np.ndarray
        분할 전/후 레이블 배열.
    feature_node : str | None
        분할에 사용된 단일 feature 노드 ID.
    feature_nodes : list[str] | None
        분할에 사용된 여러 feature 노드 ID.
    graph_scorer : GraphTaxonomyScorer | None
        계층 정보 계산기. None이면 기본 IG 반환.
    base_criterion : str
        기본 IG 계산 기준 ("entropy" | "gini").
    class_weights : dict | None
        클래스 가중치.
    alpha : float | None
        ATI 가중치 오버라이드. None이면 scorer 기본값 사용.

    Returns
    -------
    float
        PIG 값.
    """
    ig = _info_gain_base(parent_labels, left_labels, right_labels, base_criterion, class_weights)

    if graph_scorer is None:
        return ig

    # feature_nodes 구성
    nodes = []
    if feature_nodes:
        nodes = list(feature_nodes)
    elif feature_node:
        nodes = [feature_node]

    if not nodes:
        return ig

    # ATI 계산
    ati = graph_scorer.compute_ati_for_feature_nodes(nodes)

    # PIG 계산
    _alpha = alpha if alpha is not None else getattr(graph_scorer, "alpha", 1.0)
    penalty_factor = math.log(1.0 + _alpha * ati)
    pig = ig * (1.0 + penalty_factor)

    return pig


def semantic_sim_gain(
    parent_labels: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    left_feature_nodes: Optional[List[str]] = None,
    right_feature_nodes: Optional[List[str]] = None,
    graph_scorer=None,
    base_criterion: str = "entropy",
    class_weights: Optional[Dict[int, float]] = None,
    semantic_weight: float = 0.3,
) -> float:
    """Semantic Similarity 기반 분할 점수.

    Score = (1 - w) × IG + w × Sim(A)

    Sim(A) = Σ_u p_u × Sim(a_u)
    각 부분집합의 intra-similarity를 크기 비율로 가중 합산한다.

    Parameters
    ----------
    parent_labels, left_labels, right_labels : np.ndarray
        분할 전/후 레이블 배열.
    left_feature_nodes : list[str] | None
        왼쪽 부분집합의 관련 노드들.
    right_feature_nodes : list[str] | None
        오른쪽 부분집합의 관련 노드들.
    graph_scorer : GraphTaxonomyScorer | None
        계층 정보 계산기.
    base_criterion : str
        기본 IG 기준.
    class_weights : dict | None
        클래스 가중치.
    semantic_weight : float
        Sim 가중치 (0~1). 기본 0.3.

    Returns
    -------
    float
        결합 점수.
    """
    ig = _info_gain_base(parent_labels, left_labels, right_labels, base_criterion, class_weights)

    if graph_scorer is None or (not left_feature_nodes and not right_feature_nodes):
        return ig

    total = len(left_labels) + len(right_labels)
    if total == 0:
        return ig

    p_left = len(left_labels) / total
    p_right = len(right_labels) / total

    sim_left = graph_scorer.compute_nodes_intra_similarity(left_feature_nodes or [])
    sim_right = graph_scorer.compute_nodes_intra_similarity(right_feature_nodes or [])

    sim_a = p_left * sim_left + p_right * sim_right

    # 결합: IG와 Sim(A)를 가중 합산
    score = (1.0 - semantic_weight) * ig + semantic_weight * sim_a * max(ig, 0.001)

    return score


def pig_semantic_combined_gain(
    parent_labels: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    feature_node: Optional[str] = None,
    feature_nodes: Optional[List[str]] = None,
    left_feature_nodes: Optional[List[str]] = None,
    right_feature_nodes: Optional[List[str]] = None,
    graph_scorer=None,
    base_criterion: str = "entropy",
    class_weights: Optional[Dict[int, float]] = None,
    alpha: Optional[float] = None,
    semantic_weight: float = 0.2,
) -> float:
    """PIG + Semantic Similarity 결합 점수.

    Score = (1 - w) × PIG + w × Sim(A) × IG_scale

    Parameters
    ----------
    Combined parameters of pig_gain and semantic_sim_gain.

    Returns
    -------
    float
        통합 점수.
    """
    ig = _info_gain_base(parent_labels, left_labels, right_labels, base_criterion, class_weights)

    # PIG 부분
    if graph_scorer is not None:
        nodes = list(feature_nodes or [])
        if feature_node and feature_node not in nodes:
            nodes.append(feature_node)
        if nodes:
            ati = graph_scorer.compute_ati_for_feature_nodes(nodes)
            _alpha = alpha if alpha is not None else getattr(graph_scorer, "alpha", 1.0)
            pf = math.log(1.0 + _alpha * ati)
            pig = ig * (1.0 + pf)
        else:
            pig = ig
    else:
        pig = ig

    # Semantic Sim 부분
    if graph_scorer is not None and (left_feature_nodes or right_feature_nodes):
        total = len(left_labels) + len(right_labels)
        if total > 0:
            p_left = len(left_labels) / total
            p_right = len(right_labels) / total
            sim_left = graph_scorer.compute_nodes_intra_similarity(left_feature_nodes or [])
            sim_right = graph_scorer.compute_nodes_intra_similarity(right_feature_nodes or [])
            sim_a = p_left * sim_left + p_right * sim_right
        else:
            sim_a = 0.0
    else:
        sim_a = 0.0

    combined = (1.0 - semantic_weight) * pig + semantic_weight * sim_a * max(ig, 0.001)
    return combined
