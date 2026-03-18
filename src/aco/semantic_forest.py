"""
SemanticForest — ACO 기반 의미론적 앙상블 분류기

핵심 아키텍처:
    1. **부트스트랩 배깅**: 각 "트리"는 원본 데이터의 부트스트랩 샘플로 학습
    2. **엘리트 페로몬 갱신**: 상위 N% 경로만 페로몬 보강, 전역 증발 적용
    3. **다수결 투표 예측**: 모든 트리의 규칙을 앙상블하여 다수결/확률 평균
    4. **해석 모듈**: TOP-K 최고 페로몬 온톨로지 경로 추출

사용 예:
    >>> forest = SemanticForest(graph, n_trees=10, n_ants_per_tree=30)
    >>> forest.fit(feature_df, labels)
    >>> preds = forest.predict(feature_df)
    >>> forest.interpret(top_k=5)
"""

from __future__ import annotations

import copy
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .rule_extraction import (
    DecisionPath,
    RuleExtractionEngine,
    SplitCondition,
    _majority_class,
    compute_class_weights,
)
from src.utils.rule_io import RuleExporter

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class PheromonePathInfo:
    """TOP-K 해석을 위한 개별 경로 정보.

    Attributes
    ----------
    rule_string : str
        IF-THEN 규칙 문자열.
    avg_pheromone : float
        경로 엣지의 평균 페로몬 수준.
    max_pheromone : float
        경로 엣지의 최대 페로몬 수준.
    fitness : float
        경로의 적합도 점수.
    accuracy : float
        규칙의 정확도.
    coverage : int
        규칙이 커버하는 샘플 수.
    ontology_path : List[str]
        온톨로지 그래프 상의 원시 경로 (노드 ID 시퀀스).
    tree_index : int
        이 규칙이 속한 트리 인덱스.
    """

    rule_string: str = ""
    avg_pheromone: float = 0.0
    max_pheromone: float = 0.0
    fitness: float = 0.0
    accuracy: float = 0.0
    coverage: int = 0
    ontology_path: List[str] = field(default_factory=list)
    tree_index: int = -1


@dataclass
class TreeResult:
    """단일 트리(부트스트랩)의 학습 결과.

    Attributes
    ----------
    rules : List[DecisionPath]
        이 트리에서 추출한 규칙 리스트.
    oob_indices : np.ndarray
        Out-of-Bag 샘플 인덱스.
    oob_accuracy : float
        OOB 정확도 (-1이면 미평가).
    bootstrap_indices : np.ndarray
        부트스트랩 샘플 인덱스.
    """

    rules: List[DecisionPath] = field(default_factory=list)
    oob_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    oob_accuracy: float = -1.0
    bootstrap_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SemanticForest 핵심 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class SemanticForest:
    """ACO 기반 Semantic Forest 앙상블 분류기.

    Parameters
    ----------
    graph : nx.DiGraph
        OWLGraphBuilder → SMILESOntologyBridge 를 거친 온톨로지 그래프.
    n_trees : int
        앙상블의 트리(부트스트랩) 수. 기본 10.
    n_ants_per_tree : int
        각 트리 학습 시 탐색할 개미 수. 기본 30.
    elite_ratio : float
        엘리트 비율 — 상위 N% 경로만 페로몬 강화. 기본 0.2 (20%).
    evaporation_rate : float
        세대 간 전역 페로몬 증발 계수 ρ. 기본 0.1.
        τ_new = (1 - ρ) · τ_old
    min_pheromone : float
        페로몬 하한값 (완전 소멸 방지). 기본 0.01.
    max_pheromone : float
        페로몬 상한값 (조기 수렴 방지). 기본 50.0.
    bootstrap_ratio : float
        부트스트랩 샘플 비율. 기본 1.0 (원본과 동일 크기).
    alpha : float
        ACO 페로몬 지수. 기본 1.0.
    beta : float
        ACO 휴리스틱 지수. 기본 2.0.
    max_path_length : int
        규칙 최대 깊이. 기본 5.
    max_steps : int
        개미 최대 이동 횟수. 기본 80.
    min_gain : float
        정보 이득 하한. 기본 0.005.
    min_samples_leaf : int
        분할 후 최소 리프 샘플 수. 기본 10.
    criterion : str
        분할 기준. ``"entropy"`` | ``"gini"`` | ``"pig"`` | ``"semantic_similarity"`` | ``"pig_semantic"``.
        기본 ``"entropy"``.
    jump_penalty_base : float
        계층 건너뛰기 휴리스틱 감가율 p (0 <= p <= 1). 기본 0.9.
    jump_gamma : float
        계층 건너뛰기 페널티 강도 γ. 기본 1.0.
        γ=0이면 페널티 없음, γ→∞이면 건너뛰기 억제.
    pig_alpha : float
        PIG의 ATI 가중치 α. 기본 1.0.
    semantic_weight : float
        Semantic Similarity 가중치 (0~1). 기본 0.3.
    n_generations : int
        세대 수 — 페로몬 갱신 반복 횟수. 기본 3.
    seed : int | None
        재현성 시드.

    Examples
    --------
    >>> forest = SemanticForest(G, n_trees=10, n_ants_per_tree=30)
    >>> forest.fit(feature_df, pd.Series(labels))
    >>> predictions = forest.predict(feature_df)
    >>> interpretation = forest.interpret(top_k=5)
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        *,
        n_trees: int = 10,
        n_ants_per_tree: int = 30,
        elite_ratio: float = 0.2,
        evaporation_rate: float = 0.1,
        min_pheromone: float = 0.01,
        max_pheromone: float = 50.0,
        bootstrap_ratio: float = 1.0,
        alpha: float = 1.0,
        beta: float = 2.0,
        max_path_length: int = 5,
        max_steps: int = 80,
        min_gain: float = 0.005,
        min_samples_leaf: int = 10,
        criterion: str = "entropy",
        jump_penalty_base: float = 0.9,
        jump_gamma: float = 1.0,
        pig_alpha: float = 1.0,
        semantic_weight: float = 0.3,
        compute_backend: str = "auto",
        torch_device: str = "auto",
        n_generations: int = 3,
        seed: Optional[int] = None,
        task: str = "classification",
        fixed_rules: Optional[List[DecisionPath]] = None,
    ) -> None:
        self.graph = graph
        self.n_trees = n_trees
        self.n_ants_per_tree = n_ants_per_tree
        self.elite_ratio = elite_ratio
        self.evaporation_rate = evaporation_rate
        self.min_pheromone = min_pheromone
        self.max_pheromone = max_pheromone
        self.bootstrap_ratio = bootstrap_ratio
        self.alpha = alpha
        self.beta = beta
        self.max_path_length = max_path_length
        self.max_steps = max_steps
        self.min_gain = min_gain
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.jump_penalty_base = jump_penalty_base
        self.jump_gamma = jump_gamma
        self.pig_alpha = pig_alpha
        self.semantic_weight = semantic_weight
        self.compute_backend = compute_backend
        self.torch_device = torch_device
        self.n_generations = n_generations
        self.seed = seed
        self.task = task
        self.fixed_rules = fixed_rules or []

        self._rng = np.random.RandomState(seed)

        # 학습 결과 저장
        self.trees_: List[TreeResult] = []
        self.all_rules_: List[DecisionPath] = []
        self.is_fitted_: bool = False

        # 학습 데이터 참조 (predict 시 default prediction에 사용)
        self._default_prediction: Any = None
        self._feature_columns: List[str] = []
        self._class_weights: Optional[Dict[int, float]] = None

        logger.info(
            "SemanticForest initialized: n_trees=%d, n_ants=%d, "
            "elite_ratio=%.2f, evaporation=%.2f, generations=%d",
            n_trees,
            n_ants_per_tree,
            elite_ratio,
            evaporation_rate,
            n_generations,
        )
        if str(self.compute_backend).lower() != "auto" or str(self.torch_device).lower() != "auto":
            logger.info(
                "Backend options received: compute_backend=%s, torch_device=%s "
                "(current ACO path uses numpy execution)",
                self.compute_backend,
                self.torch_device,
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 공개 API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def fit(
        self,
        feature_df: pd.DataFrame,
        labels: pd.Series,
    ) -> "SemanticForest":
        """앙상블을 학습한다.

        1. 세대 반복 (n_generations)
            a. 각 트리에 대해 부트스트랩 샘플링
            b. RuleExtractionEngine 으로 규칙 추출
            c. 모든 규칙 수집 후 엘리트 페로몬 갱신
        2. 최종 규칙 집합 정렬 및 저장

        Parameters
        ----------
        feature_df : pd.DataFrame
            학습 데이터. 각 컬럼 = 특성, 각 행 = 샘플.
        labels : pd.Series
            학습 라벨.

        Returns
        -------
        self
        """
        n_samples = len(labels)
        label_array = labels.values.copy()
        if self.task == "regression":
            self._default_prediction = float(np.mean(label_array.astype(float)))
        else:
            self._default_prediction = int(_majority_class(label_array))
        self._feature_columns = list(feature_df.columns)

        # ── Class Weight 자동 계산 (Mode Collapse 방지, classification only) ──
        if self.task == "regression":
            self._class_weights = None
        else:
            self._class_weights = compute_class_weights(label_array)
            logger.info(
                "Class weights computed: %s",
                {k: f"{v:.3f}" for k, v in self._class_weights.items()},
            )

        logger.info(
            "fit() started: %d samples × %d features, %d trees × %d ants, "
            "%d generations",
            n_samples,
            len(self._feature_columns),
            self.n_trees,
            self.n_ants_per_tree,
            self.n_generations,
        )

        if self.fixed_rules:
            logger.info("Injecting %d fixed rules to pheromones.", len(self.fixed_rules))
            for rule in self.fixed_rules:
                raw = rule.raw_path
                if len(raw) < 2:
                    continue
                delta_tau = rule.fitness / len(raw) if rule.fitness > 0 else 1.0 / len(raw)
                for i in range(len(raw) - 1):
                    u, v = raw[i], raw[i + 1]
                    if self.graph.has_edge(u, v):
                        self.graph.edges[u, v]["pheromone"] = min(
                            self.graph.edges[u, v].get("pheromone", 1.0) + delta_tau,
                            self.max_pheromone
                        )
                    if self.graph.has_edge(v, u):
                        self.graph.edges[v, u]["pheromone"] = min(
                            self.graph.edges[v, u].get("pheromone", 1.0) + delta_tau,
                            self.max_pheromone
                        )

        # ── 세대 반복 ──
        best_trees: List[TreeResult] = []

        for gen in range(self.n_generations):
            logger.info("=== Generation %d/%d ===", gen + 1, self.n_generations)

            gen_rules: List[DecisionPath] = []
            gen_trees: List[TreeResult] = []

            for t in range(self.n_trees):
                # ── 부트스트랩 샘플링 ──
                bs_size = max(1, int(n_samples * self.bootstrap_ratio))
                bs_indices = self._rng.choice(n_samples, size=bs_size, replace=True)
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[np.unique(bs_indices)] = False
                oob_indices = np.where(oob_mask)[0]

                bs_feature_df = feature_df.iloc[bs_indices].reset_index(drop=True)
                bs_labels = pd.Series(label_array[bs_indices])

                # ── RuleExtractionEngine 으로 규칙 추출 ──
                # 엔진에 페로몬이 누적된 공유 그래프를 전달
                engine = RuleExtractionEngine(
                    graph=self.graph,
                    feature_df=bs_feature_df,
                    label_series=bs_labels,
                    alpha=self.alpha,
                    beta=self.beta,
                    max_path_length=self.max_path_length,
                    max_steps=self.max_steps,
                    min_gain=self.min_gain,
                    min_samples_leaf=self.min_samples_leaf,
                    criterion=self.criterion,
                    jump_penalty_base=self.jump_penalty_base,
                    jump_gamma=self.jump_gamma,
                    pig_alpha=self.pig_alpha,
                    semantic_weight=self.semantic_weight,
                    seed=(self.seed + gen * self.n_trees + t)
                    if self.seed is not None
                    else None,
                    class_weights=self._class_weights,
                    task=self.task,
                )

                rules = engine.extract_rules(
                    n_ants=self.n_ants_per_tree, deduplicate=True
                )

                # ── OOB 평가 ──
                oob_acc = -1.0
                if len(oob_indices) > 0 and len(rules) > 0:
                    oob_df = feature_df.iloc[oob_indices].reset_index(drop=True)
                    oob_labels = label_array[oob_indices]
                    eval_result = engine.evaluate_rules(
                        rules, oob_df, oob_labels
                    )
                    oob_acc = eval_result["avg_accuracy"]

                tree_result = TreeResult(
                    rules=rules,
                    oob_indices=oob_indices,
                    oob_accuracy=oob_acc,
                    bootstrap_indices=np.unique(bs_indices),
                )
                gen_trees.append(tree_result)
                gen_rules.extend(rules)

                logger.info(
                    "  Tree %d/%d: %d rules, OOB accuracy=%.4f",
                    t + 1,
                    self.n_trees,
                    len(rules),
                    oob_acc,
                )

            # ── 엘리트 페로몬 갱신 ──
            self._elite_pheromone_update(gen_rules)
            best_trees = gen_trees

            # 세대 요약
            gen_rule_count = sum(len(tr.rules) for tr in gen_trees)
            avg_oob = np.mean(
                [tr.oob_accuracy for tr in gen_trees if tr.oob_accuracy >= 0]
            ) if any(tr.oob_accuracy >= 0 for tr in gen_trees) else 0.0
            logger.info(
                "  Generation %d summary: %d total rules, avg OOB=%.4f",
                gen + 1,
                gen_rule_count,
                avg_oob,
            )

        # ── 최종 저장 ──
        if self.fixed_rules:
            dummy_tree_result = TreeResult(
                rules=copy.deepcopy(self.fixed_rules),
                oob_accuracy=1.0,
            )
            best_trees.append(dummy_tree_result)

        self.trees_ = best_trees
        self.all_rules_ = []
        for tr in self.trees_:
            self.all_rules_.extend(tr.rules)

        # 적합도 내림차순 정렬
        self.all_rules_.sort(key=lambda p: p.fitness, reverse=True)

        # 중복 제거
        self.all_rules_ = self._deduplicate_rules(self.all_rules_)

        self.is_fitted_ = True
        logger.info(
            "fit() complete: %d trees, %d unique rules total",
            len(self.trees_),
            len(self.all_rules_),
        )
        return self

    def predict(
        self,
        feature_df: pd.DataFrame,
        *,
        method: str = "majority",
    ) -> np.ndarray:
        """앙상블 예측을 수행한다.

        Parameters
        ----------
        feature_df : pd.DataFrame
            예측 대상 데이터.
        method : str
            ``"majority"`` — 다수결 투표.
            ``"weighted"`` — 적합도 가중 투표.

        Returns
        -------
        np.ndarray
            예측 라벨 배열.
        """
        if not self.is_fitted_:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        n_samples = len(feature_df)
        predictions = np.full(n_samples, fill_value=self._default_prediction)

        if self.task == "regression":
            predictions = self._predict_regression_mean(feature_df)
        elif method == "majority":
            predictions = self._predict_majority(feature_df)
        elif method == "weighted":
            predictions = self._predict_weighted(feature_df)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'majority' or 'weighted'.")

        return predictions

    def export_rules(self, filepath: str, format: str = "json", top_k: Optional[int] = None) -> None:
        """현재 학습된 규칙을 저장한다.

        Parameters
        ----------
        filepath : str
            저장할 파일 경로.
        format : str
            'json', 'csv', 'txt', 'markdown' 지원.
        top_k : int | None
            상위 K개 규칙만 저장.
        """
        if not self.is_fitted_:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        rules_to_export = self.all_rules_
        if top_k is not None:
            rules_to_export = rules_to_export[:top_k]

        fmt = format.lower()
        if fmt == "json":
            RuleExporter.export_to_json(rules_to_export, filepath)
        elif fmt == "csv":
            RuleExporter.export_to_csv(rules_to_export, filepath)
        elif fmt in ("md", "markdown"):
            RuleExporter.export_to_markdown(rules_to_export, filepath)
        elif fmt == "txt":
            RuleExporter.export_to_txt(rules_to_export, filepath)
        else:
            raise ValueError(f"Unknown format: {format}")

    def predict_proba(
        self,
        feature_df: pd.DataFrame,
    ) -> np.ndarray:
        """리프 노드 클래스 비율 기반 확률 추정.

        각 트리별로 매칭되는 규칙의 리프 노드에 도달한 샘플들의
        클래스 비율(leaf_probability)을 확률로 사용하고,
        숲 전체 트리의 확률값을 평균 내어 반환한다.

        단순 0/1 다수결을 넘어서, 연속적인 확률 분포를 반환하므로
        AUC-ROC 평가에 최적화된다.

        Parameters
        ----------
        feature_df : pd.DataFrame
            예측 대상 데이터.

        Returns
        -------
        np.ndarray
            shape = (n_samples, 2). [:, 0] = P(class=0), [:, 1] = P(class=1).
        """
        if not self.is_fitted_:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        n_samples = len(feature_df)

        # 각 트리별로 리프 확률을 수집
        proba_accum = np.zeros((n_samples, 2), dtype=float)
        n_contributing_trees = np.zeros(n_samples, dtype=float)

        for tree_result in self.trees_:
            tree_proba = self._get_tree_leaf_proba(feature_df, tree_result.rules)
            # tree_proba: (n_samples, 2), 미커버 샘플은 [-1, -1]

            for i in range(n_samples):
                if tree_proba[i, 0] >= 0:  # 유효한 예측
                    proba_accum[i] += tree_proba[i]
                    n_contributing_trees[i] += 1.0

        # 평균 확률 계산
        result = np.zeros((n_samples, 2), dtype=float)
        for i in range(n_samples):
            if n_contributing_trees[i] > 0:
                result[i] = proba_accum[i] / n_contributing_trees[i]
            else:
                # 미커버: base rate로 fallback
                result[i, 0] = 1.0 if self._default_prediction == 0 else 0.0
                result[i, 1] = 1.0 if self._default_prediction == 1 else 0.0

        # 확률 합이 1이 되도록 재정규화
        row_sums = result.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        result = result / row_sums

        return result

    def _get_tree_leaf_proba(
        self,
        feature_df: pd.DataFrame,
        rules: List[DecisionPath],
    ) -> np.ndarray:
        """단일 트리의 규칙에서 리프 확률을 추출한다.

        각 샘플에 대해 매칭되는 첫 번째 규칙의 leaf_class_dist를
        기반으로 확률을 계산한다.

        Returns
        -------
        np.ndarray
            shape = (n_samples, 2). 미커버 = [-1, -1].
        """
        n_samples = len(feature_df)
        proba = np.full((n_samples, 2), -1.0, dtype=float)

        for idx in range(n_samples):
            for rule in rules:
                # 이 규칙이 이 샘플을 커버하는지 확인
                match = True
                for cond in rule.conditions:
                    col = cond.feature
                    if col not in feature_df.columns:
                        match = False
                        break
                    val = feature_df.iloc[idx][col]
                    if not cond.evaluate(val):
                        match = False
                        break

                if match:
                    # 리프 노드의 클래스 비율 기반 확률
                    # class_weights가 있으면 가중 확률로 변환하여
                    # 소수 클래스 예측이 threshold 0.5를 넘을 수 있게 한다.
                    if rule.leaf_class_dist:
                        total = sum(rule.leaf_class_dist.values())
                        if total > 0 and self._class_weights:
                            # 가중 확률: count × weight
                            w0 = (rule.leaf_class_dist.get(0, 0)
                                  * self._class_weights.get(0, 1.0))
                            w1 = (rule.leaf_class_dist.get(1, 0)
                                  * self._class_weights.get(1, 1.0))
                            total_w = w0 + w1
                            if total_w > 0:
                                proba[idx, 0] = w0 / total_w
                                proba[idx, 1] = w1 / total_w
                            else:
                                proba[idx, 0] = 0.5
                                proba[idx, 1] = 0.5
                        elif total > 0:
                            p0 = rule.leaf_class_dist.get(0, 0) / total
                            p1 = rule.leaf_class_dist.get(1, 0) / total
                            proba[idx, 0] = p0
                            proba[idx, 1] = p1
                        else:
                            proba[idx, 0] = 0.5
                            proba[idx, 1] = 0.5
                    else:
                        # leaf_class_dist가 없으면 hard prediction fallback
                        pred = rule.prediction
                        proba[idx, 0] = 1.0 if pred == 0 else 0.0
                        proba[idx, 1] = 1.0 if pred == 1 else 0.0
                    break  # 첫 번째 매칭 규칙 사용

        return proba

    def interpret(
        self,
        top_k: int = 5,
    ) -> List[PheromonePathInfo]:
        """TOP-K 최고 페로몬 온톨로지 경로를 추출하여 해석 정보를 반환.

        각 규칙의 원시 경로(raw_path)를 따라 현재 그래프의 페로몬 수준을
        측정하고, 평균 페로몬이 가장 높은 상위 K개 경로를 반환한다.

        Parameters
        ----------
        top_k : int
            반환할 상위 경로 수. 기본 5.

        Returns
        -------
        List[PheromonePathInfo]
            최고 페로몬 경로 정보 리스트 (내림차순 정렬).
        """
        if not self.is_fitted_:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        G = self.graph
        path_infos: List[PheromonePathInfo] = []

        for tree_idx, tree_result in enumerate(self.trees_):
            for rule in tree_result.rules:
                if len(rule.raw_path) < 2:
                    continue

                # 경로 엣지의 페로몬 수집
                pheromones = []
                for i in range(len(rule.raw_path) - 1):
                    u, v = rule.raw_path[i], rule.raw_path[i + 1]
                    if G.has_edge(u, v):
                        pheromones.append(
                            G.edges[u, v].get("pheromone", 1.0)
                        )
                    elif G.has_edge(v, u):
                        pheromones.append(
                            G.edges[v, u].get("pheromone", 1.0)
                        )
                    else:
                        pheromones.append(0.0)

                if not pheromones:
                    continue

                info = PheromonePathInfo(
                    rule_string=rule.to_rule_string(),
                    avg_pheromone=float(np.mean(pheromones)),
                    max_pheromone=float(np.max(pheromones)),
                    fitness=rule.fitness,
                    accuracy=rule.accuracy,
                    coverage=rule.coverage,
                    ontology_path=list(rule.raw_path),
                    tree_index=tree_idx,
                )
                path_infos.append(info)

        # 평균 페로몬 내림차순 → 적합도 내림차순 정렬
        path_infos.sort(
            key=lambda p: (p.avg_pheromone, p.fitness), reverse=True
        )

        return path_infos[:top_k]

    def get_feature_importance(self) -> Dict[str, float]:
        """규칙에 등장하는 특성별 중요도를 반환한다.

        중요도 = Σ(규칙의 fitness × 해당 특성 IG) / 전체 규칙 수

        Returns
        -------
        dict[str, float]
            특성명 → 중요도 점수. 내림차순 정렬.
        """
        if not self.is_fitted_:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        importance: Dict[str, float] = {}
        count: Dict[str, int] = {}

        for rule in self.all_rules_:
            for cond in rule.conditions:
                feat = cond.feature
                score = rule.fitness * cond.info_gain
                importance[feat] = importance.get(feat, 0.0) + score
                count[feat] = count.get(feat, 0) + 1

        # 규칙 수로 정규화
        n_rules = max(len(self.all_rules_), 1)
        for feat in importance:
            importance[feat] /= n_rules

        # 내림차순 정렬
        return dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)
        )

    def summary(self) -> Dict[str, Any]:
        """학습 결과 요약 딕셔너리.

        Returns
        -------
        dict
            n_trees, total_rules, avg_oob_accuracy, pheromone_stats, ...
        """
        if not self.is_fitted_:
            return {"is_fitted": False}

        oob_accs = [
            tr.oob_accuracy for tr in self.trees_ if tr.oob_accuracy >= 0
        ]
        rules_per_tree = [len(tr.rules) for tr in self.trees_]

        # 그래프 페로몬 통계
        all_pher = [
            edata.get("pheromone", 1.0)
            for _, _, edata in self.graph.edges(data=True)
        ]

        return {
            "is_fitted": True,
            "n_trees": len(self.trees_),
            "total_unique_rules": len(self.all_rules_),
            "rules_per_tree": {
                "mean": float(np.mean(rules_per_tree)) if rules_per_tree else 0,
                "min": int(np.min(rules_per_tree)) if rules_per_tree else 0,
                "max": int(np.max(rules_per_tree)) if rules_per_tree else 0,
            },
            "avg_oob_accuracy": float(np.mean(oob_accs)) if oob_accs else -1.0,
            "pheromone_stats": {
                "mean": float(np.mean(all_pher)) if all_pher else 0.0,
                "std": float(np.std(all_pher)) if all_pher else 0.0,
                "min": float(np.min(all_pher)) if all_pher else 0.0,
                "max": float(np.max(all_pher)) if all_pher else 0.0,
            },
            "feature_importance": self.get_feature_importance(),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 내부 구현 — 페로몬 갱신
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _elite_pheromone_update(
        self,
        rules: List[DecisionPath],
    ) -> None:
        """엘리트 기반 페로몬 갱신 알고리즘 (F1-score 기반).

        τ_ij = (1 − ρ) · τ_ij + Δτ_ij   (엘리트 경로만)

        Fitness가 F1-score 기반으로 변경되어:
        - 전부 0 예측 트리: Recall=0 → F1=0 → fitness=0 → 페로몬 증발만 발생
        - 소수 클래스 올바르게 분류: F1>0 → 페로몬 강화

        Steps:
            1. 전역 증발: 모든 엣지에 (1 - ρ) 적용
            2. 엘리트 선택: F1-based fitness 상위 elite_ratio 경로만 선택
            3. 페로몬 보강: Δτ = fitness / path_length
            4. 클램핑: [min_pheromone, max_pheromone] 범위 유지

        Parameters
        ----------
        rules : List[DecisionPath]
            현재 세대에서 생성된 모든 규칙.
        """
        G = self.graph
        rho = self.evaporation_rate

        if not rules:
            return

        # ── Step 1: 전역 증발 ──
        for u, v, edata in G.edges(data=True):
            old_tau = edata.get("pheromone", 1.0)
            new_tau = old_tau * (1.0 - rho)
            edata["pheromone"] = max(new_tau, self.min_pheromone)

        # ── Step 2: 엘리트 선택 (F1-based fitness) ──
        # fitness는 이제 F1-score 기반이므로, Mode Collapse(F1=0) 경로는 자동 탈락
        sorted_rules = sorted(rules, key=lambda r: r.fitness, reverse=True)
        n_elite = max(1, int(len(sorted_rules) * self.elite_ratio))
        elite_rules = sorted_rules[:n_elite]

        # 엘리트 통계 로깅
        elite_f1s = [r.f1_score for r in elite_rules]
        avg_elite_f1 = float(np.mean(elite_f1s)) if elite_f1s else 0.0
        nonzero_f1 = sum(1 for f in elite_f1s if f > 0)

        logger.info(
            "Elite pheromone update: %d/%d rules selected (top %.0f%%), "
            "avg_elite_F1=%.4f, nonzero_F1=%d/%d",
            n_elite,
            len(rules),
            self.elite_ratio * 100,
            avg_elite_f1,
            nonzero_f1,
            n_elite,
        )

        # ── Step 3: 엘리트 경로 페로몬 보강 ──
        for rule in elite_rules:
            raw = rule.raw_path
            if len(raw) < 2:
                continue

            # Δτ = fitness / path_length (fitness는 F1 기반)
            delta_tau = rule.fitness / len(raw)

            for i in range(len(raw) - 1):
                u, v = raw[i], raw[i + 1]
                if G.has_edge(u, v):
                    G.edges[u, v]["pheromone"] += delta_tau
                if G.has_edge(v, u):
                    G.edges[v, u]["pheromone"] += delta_tau

        # ── Step 4: 클램핑 ──
        for u, v, edata in G.edges(data=True):
            edata["pheromone"] = max(
                self.min_pheromone,
                min(edata["pheromone"], self.max_pheromone),
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 내부 구현 — 예측
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _apply_tree_rules(
        self,
        feature_df: pd.DataFrame,
        rules: List[DecisionPath],
    ) -> np.ndarray:
        """단일 트리의 규칙 리스트를 사용하여 예측.

        규칙은 적합도 순서대로 매칭을 시도한다.
        매칭되는 규칙이 없으면 -999 (미커버 마커).

        Returns
        -------
        np.ndarray
            예측 배열. 미커버 = -999.
        """
        n_samples = len(feature_df)
        preds = np.full(n_samples, fill_value=-999, dtype=float)

        for idx in range(n_samples):
            row = {
                col: feature_df.iloc[idx][col] for col in feature_df.columns
            }
            for rule in rules:
                pred = rule.evaluate_sample(row)
                if pred is not None:
                    preds[idx] = pred
                    break

        return preds

    def _predict_majority(self, feature_df: pd.DataFrame) -> np.ndarray:
        """리프 확률 기반 다수결 예측.

        predict_proba()로 확률을 구한 뒤 threshold=0.5로 분류한다.
        이전의 hard-vote 방식 대비 리프 노드 비율을 반영하므로
        mode collapse를 방지한다.
        """
        proba = self.predict_proba(feature_df)
        # proba[:, 1] >= 0.5 이면 class 1, 아니면 class 0
        preds = (proba[:, 1] >= 0.5).astype(int)
        return preds

    def _predict_weighted(self, feature_df: pd.DataFrame) -> np.ndarray:
        """적합도 가중 리프 확률 예측.

        각 트리의 OOB 정확도를 가중치로 사용하여 리프 확률을 가중 평균한다.
        """
        n_samples = len(feature_df)

        proba_accum = np.zeros((n_samples, 2), dtype=float)
        weight_accum = np.zeros(n_samples, dtype=float)

        for tree_result in self.trees_:
            tree_proba = self._get_tree_leaf_proba(feature_df, tree_result.rules)
            weight = max(tree_result.oob_accuracy, 0.01)

            for i in range(n_samples):
                if tree_proba[i, 0] >= 0:  # 유효한 예측
                    proba_accum[i] += tree_proba[i] * weight
                    weight_accum[i] += weight

        final_preds = np.full(n_samples, fill_value=self._default_prediction)
        for i in range(n_samples):
            if weight_accum[i] > 0:
                avg_p1 = proba_accum[i, 1] / weight_accum[i]
                final_preds[i] = 1 if avg_p1 >= 0.5 else 0
            else:
                final_preds[i] = self._default_prediction

        return final_preds

    def _predict_regression_mean(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Regression prediction: average of per-tree rule predictions."""
        n_samples = len(feature_df)
        pred_accum = np.zeros(n_samples, dtype=float)
        n_contributing = np.zeros(n_samples, dtype=float)

        for tree_result in self.trees_:
            tree_preds = self._apply_tree_rules(feature_df, tree_result.rules)
            for i in range(n_samples):
                if tree_preds[i] != -999:
                    pred_accum[i] += tree_preds[i]
                    n_contributing[i] += 1.0

        result = np.full(n_samples, fill_value=self._default_prediction, dtype=float)
        for i in range(n_samples):
            if n_contributing[i] > 0:
                result[i] = pred_accum[i] / n_contributing[i]

        return result

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 유틸리티
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _deduplicate_rules(rules: List[DecisionPath]) -> List[DecisionPath]:
        """동일 조건 조합의 중복 규칙 제거."""
        seen: set = set()
        unique: List[DecisionPath] = []
        for r in rules:
            key = "|".join(
                f"{c.feature}{c.operator}{c.threshold:.4f}"
                for c in r.conditions
            )
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"SemanticForest(n_trees={self.n_trees}, "
            f"n_ants={self.n_ants_per_tree}, "
            f"elite={self.elite_ratio:.0%}, "
            f"status={status})"
        )
