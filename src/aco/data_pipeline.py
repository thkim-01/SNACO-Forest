"""
ToxicityDataPipeline - 6대 독성 데이터셋 통합 파이프라인

데이터 로드 → SMILES 디스크립터 계산 → 온톨로지 라우팅 →
Feature→Ontology 브릿지 → 도메인 휴리스틱 적용 → 계층 게이트 설치
→ SemanticForest 학습까지의 전 과정을 자동화한다.

Usage:
    pipeline = ToxicityDataPipeline("configs/dataset_ontology_config.json")
    result = pipeline.run("bbbp")
    result = pipeline.run("tox21", target="NR-AhR")
    results = pipeline.run_all()
"""

from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from .knowledge_heuristics import KnowledgeHeuristics
from .hierarchical_search import HierarchicalSearch
from .ontology_router import OntologyRouter
from .semantic_forest import SemanticForest
from .smiles_descriptor import compute_descriptors_batch
from .smiles_ontology_bridge import SMILESOntologyBridge

logger = logging.getLogger(__name__)

DATASET_EXECUTION_ORDER = [
    "bbbp",
    "bace",
    "hiv",
    "clintox",
    "tox21",
    "sider",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터셋 로더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_dataset(
    dataset_name: str,
    config: Dict[str, Any],
    base_dir: Path,
    *,
    target: Optional[str] = None,
    max_samples: int = 0,
) -> Tuple[List[str], np.ndarray, str]:
    """데이터셋을 로드하여 SMILES 리스트와 라벨 배열을 반환한다.

    Parameters
    ----------
    dataset_name : str
        데이터셋 이름.
    config : dict
        데이터셋 설정 딕셔너리.
    base_dir : Path
        프로젝트 루트.
    target : str | None
        멀티타겟 데이터셋의 타겟명.
    max_samples : int
        최대 샘플 수 (0=무제한).

    Returns
    -------
    (smiles_list, labels, actual_target_name)
    """
    csv_path = base_dir / config["file"]
    smiles_col = config["smiles_col"]
    default_label_col = config["label_col"]

    # 타겟 결정
    actual_target = target or default_label_col

    df = pd.read_csv(str(csv_path))

    # 멀티타겟: 지정된 타겟 컬럼 사용
    if actual_target not in df.columns:
        raise ValueError(
            f"Target '{actual_target}' not found in {csv_path.name}. "
            f"Available columns: {list(df.columns)}"
        )

    # NaN 라벨 제거
    valid_mask = df[actual_target].notna()
    df = df[valid_mask].reset_index(drop=True)

    # label을 정수로 변환
    labels = df[actual_target].astype(int).values

    # SMILES 추출
    smiles_list = df[smiles_col].tolist()

    # 샘플 수 제한
    if max_samples > 0 and len(smiles_list) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(smiles_list), size=max_samples, replace=False)
        smiles_list = [smiles_list[i] for i in indices]
        labels = labels[indices]

    logger.info(
        "Loaded %s [%s]: %d samples (pos=%d, neg=%d)",
        dataset_name,
        actual_target,
        len(labels),
        int(np.sum(labels == 1)),
        int(np.sum(labels == 0)),
    )
    return smiles_list, labels, actual_target


def stratified_split(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """계층화 랜덤 분할."""
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    train_indices: List[int] = []
    test_indices: List[int] = []

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(len(cls_idx) * test_size))
        test_indices.extend(cls_idx[:n_test].tolist())
        train_indices.extend(cls_idx[n_test:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    train_df = feature_df.iloc[train_indices].reset_index(drop=True)
    test_df = feature_df.iloc[test_indices].reset_index(drop=True)
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]

    return train_df, test_df, train_labels, test_labels


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """평가 메트릭 계산 (accuracy, balanced_accuracy, F1, precision, recall, AUC-ROC)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    accuracy = float(np.sum(y_true == y_pred) / n)

    # 이진 분류 메트릭
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)

    # Balanced accuracy
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (tpr + tnr) / 2.0

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    # AUC-ROC (순수 numpy 구현)
    if y_proba is not None:
        y_proba_arr = np.asarray(y_proba)
        unique_classes = np.unique(y_true)
        if len(unique_classes) == 2:
            try:
                if y_proba_arr.ndim == 2:
                    if y_proba_arr.shape[1] < 2:
                        raise ValueError(
                            "predict_proba returned 2D array with <2 columns"
                        )
                    pos_proba = y_proba_arr[:, 1]
                else:
                    pos_proba = y_proba_arr.reshape(-1)

                if len(pos_proba) != n:
                    raise ValueError(
                        f"Probability length mismatch: got {len(pos_proba)}, expected {n}"
                    )

                sorted_idx = np.argsort(-pos_proba)
                y_sorted = y_true[sorted_idx]
                n_pos = np.sum(y_true == 1)
                n_neg = n - n_pos
                if n_pos > 0 and n_neg > 0:
                    tpr_arr = np.cumsum(y_sorted == 1) / n_pos
                    fpr_arr = np.cumsum(y_sorted == 0) / n_neg
                    if hasattr(np, "trapezoid"):
                        auc = float(np.trapezoid(tpr_arr, fpr_arr))
                    else:
                        auc = float(np.trapz(tpr_arr, fpr_arr))
                    metrics["auc_roc"] = auc
            except Exception as e:
                logger.warning(
                    "AUC-ROC unavailable: %s (classes=%s, y_proba_shape=%s)",
                    e,
                    unique_classes.tolist(),
                    getattr(y_proba_arr, "shape", None),
                )
        else:
            logger.warning(
                "AUC-ROC skipped: y_true must contain 2 classes, got %s",
                unique_classes.tolist(),
            )

    return metrics


def validate_rule_hierarchical_inheritance(
    graph: nx.DiGraph,
    rules: List[Any],
) -> Dict[str, Any]:
    """추출 규칙이 온톨로지 계층 상속을 일관되게 추적하는지 검증한다.

    검증 항목:
    1) 동일 feature에 대한 규칙 조건의 구간 모순 여부
    2) 규칙 raw_path 내 하위→상위(subClassOf) 상속 순서 역전 여부
    """
    if not rules:
        return {
            "n_rules_checked": 0,
            "n_consistent_rules": 0,
            "consistency_rate": 1.0,
            "n_rules_with_hierarchy": 0,
            "n_rules_with_contradiction": 0,
            "n_rules_with_order_violation": 0,
            "n_inheritance_edges_in_paths": 0,
            "sample_violations": [],
        }

    # subClassOf 관계만 추린 계층 그래프 (child -> parent)
    hierarchy = nx.DiGraph()
    for u, v, edata in graph.edges(data=True):
        if edata.get("predicate") == "subClassOf":
            hierarchy.add_edge(u, v)

    ancestors_cache: Dict[str, set] = {}

    def ancestors_of(node_id: str) -> set:
        if node_id not in ancestors_cache:
            if hierarchy.has_node(node_id):
                ancestors_cache[node_id] = nx.descendants(hierarchy, node_id)
            else:
                ancestors_cache[node_id] = set()
        return ancestors_cache[node_id]

    n_consistent = 0
    n_with_hierarchy = 0
    n_with_contradiction = 0
    n_with_order_violation = 0
    n_inheritance_edges = 0
    violations: List[Dict[str, Any]] = []

    for ridx, rule in enumerate(rules):
        # ── 1) 조건 모순 검사 ──
        bounds: Dict[str, Dict[str, float]] = {}
        contradiction = False
        contradiction_feature = None
        contradiction_bounds = None

        for cond in getattr(rule, "conditions", []):
            feat = cond.feature
            if feat not in bounds:
                bounds[feat] = {"lb": float("-inf"), "ub": float("inf")}
            if cond.operator == "<=":
                bounds[feat]["ub"] = min(bounds[feat]["ub"], float(cond.threshold))
            else:
                bounds[feat]["lb"] = max(bounds[feat]["lb"], float(cond.threshold))

            if bounds[feat]["lb"] >= bounds[feat]["ub"]:
                contradiction = True
                contradiction_feature = feat
                contradiction_bounds = dict(bounds[feat])
                break

        # ── 2) 계층 상속 순서 검사 (하위가 상위보다 먼저 나타나야 함) ──
        raw_path = list(getattr(rule, "raw_path", []) or [])
        path_pos = {nid: i for i, nid in enumerate(raw_path)}
        order_violations = 0
        has_hierarchy = False

        for i in range(len(raw_path) - 1):
            u, v = raw_path[i], raw_path[i + 1]

            forward_pred = graph.edges[u, v].get("predicate") if graph.has_edge(u, v) else None
            reverse_pred = graph.edges[v, u].get("predicate") if graph.has_edge(v, u) else None

            if forward_pred in {"subClassOf", "hasSubClass"}:
                n_inheritance_edges += 1
                has_hierarchy = True

            # 비정상 순서: 현재 진행(u->v) 방향에는 상속 엣지가 없고,
            # 반대 방향(v->u)에만 상속 엣지가 있을 때
            if forward_pred not in {"subClassOf", "hasSubClass"} and reverse_pred in {"subClassOf", "hasSubClass"}:
                order_violations += 1
                has_hierarchy = True

        # path 내에 조상/자손 쌍이 존재하면 계층 사용 규칙으로 간주
        for node_id in raw_path:
            if ancestors_of(node_id).intersection(path_pos.keys()):
                has_hierarchy = True
                break

        if has_hierarchy:
            n_with_hierarchy += 1

        is_consistent = (not contradiction) and (order_violations == 0)
        if is_consistent:
            n_consistent += 1
        else:
            if contradiction:
                n_with_contradiction += 1
            if order_violations > 0:
                n_with_order_violation += 1

            if len(violations) < 20:
                violations.append({
                    "rule_index": ridx,
                    "rule": rule.to_rule_string() if hasattr(rule, "to_rule_string") else str(rule),
                    "coverage": int(getattr(rule, "coverage", 0)),
                    "contradiction": contradiction,
                    "contradiction_feature": contradiction_feature,
                    "contradiction_bounds": contradiction_bounds,
                    "order_violations": int(order_violations),
                })

    total = len(rules)
    return {
        "n_rules_checked": total,
        "n_consistent_rules": n_consistent,
        "consistency_rate": (n_consistent / total) if total > 0 else 1.0,
        "n_rules_with_hierarchy": n_with_hierarchy,
        "n_rules_with_contradiction": n_with_contradiction,
        "n_rules_with_order_violation": n_with_order_violation,
        "n_inheritance_edges_in_paths": n_inheritance_edges,
        "sample_violations": violations,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ToxicityDataPipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ToxicityDataPipeline:
    """6대 독성 데이터셋 통합 파이프라인.

    Parameters
    ----------
    config_path : str | Path
        ``configs/dataset_ontology_config.json`` 경로.
    base_dir : str | Path | None
        프로젝트 루트.

    Examples
    --------
    >>> pipeline = ToxicityDataPipeline()
    >>> result = pipeline.run("bbbp")
    >>> results = pipeline.run_all()
    """

    def __init__(
        self,
        config_path: str | Path = "configs/dataset_ontology_config.json",
        base_dir: Optional[str | Path] = None,
    ) -> None:
        self.config_path = Path(config_path)
        if base_dir is None:
            self.base_dir = self.config_path.resolve().parent.parent
        else:
            self.base_dir = Path(base_dir).resolve()

        # OntologyRouter 초기화
        self.router = OntologyRouter(
            config_path=self.config_path,
            base_dir=self.base_dir,
            use_cache=True,
        )

        self._config = self.router._config

        logger.info(
            "ToxicityDataPipeline ready: base_dir=%s, datasets=%s",
            self.base_dir,
            self.router.available_datasets,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 공개 API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def run(
        self,
        dataset_name: str,
        *,
        target: Optional[str] = None,
        ontology_override: Optional[str] = None,
        max_samples: int = 0,
        n_trees: Optional[int] = None,
        n_ants_per_tree: Optional[int] = None,
        n_generations: Optional[int] = None,
        criterion: Optional[str] = None,
        jump_penalty_base: Optional[float] = None,
        jump_gamma: Optional[float] = None,
        pig_alpha: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        seed: Optional[int] = None,
        skip_fit: bool = False,
    ) -> Dict[str, Any]:
        """단일 데이터셋에 대해 전체 파이프라인을 실행한다.

        Parameters
        ----------
        dataset_name : str
            데이터셋 이름.
        target : str | None
            멀티타겟 데이터셋의 타겟명.
        ontology_override : str | None
            데이터셋 기본 매핑 대신 강제로 사용할 온톨로지 이름.
        max_samples : int
            최대 샘플 수 (0=무제한).
        n_trees, n_ants_per_tree, n_generations, seed : int | None
            SemanticForest 파라미터 오버라이드.
        criterion : str | None
            분할 기준 오버라이드 ("entropy" | "gini" | "gain_ratio" | "chi_square"
            | "pig" | "semantic_similarity" | "pig_semantic").
        jump_penalty_base : float | None
            계층 건너뛰기 η 감가율 p (0 <= p <= 1). None이면 기본값 사용.
        jump_gamma : float | None
            계층 건너뛰기 페널티 강도 γ. 0이면 페널티 없음.
        pig_alpha : float | None
            PIG(Penalized Information Gain)의 ATI 가중치 α. 기본 1.0.
        semantic_weight : float | None
            Semantic Similarity 가중치 (0~1). 기본 0.3.
        skip_fit : bool
            True면 그래프 생성까지만 수행 (테스트/디버깅용).

        Returns
        -------
        dict
            실행 결과 딕셔너리.
        """
        t0 = time.time()
        ds_cfg = self.router.get_dataset_config(dataset_name)
        defaults = self.router.get_defaults()

        _n_trees = n_trees or defaults.get("n_trees", 8)
        _n_ants = n_ants_per_tree or defaults.get("n_ants_per_tree", 25)
        _n_gen = n_generations or defaults.get("n_generations", 3)
        _seed = seed or defaults.get("seed", 42)
        _test_size = defaults.get("test_size", 0.2)
        _criterion = criterion or defaults.get("criterion", "entropy")
        _jump_penalty_base = (
            jump_penalty_base
            if jump_penalty_base is not None
            else defaults.get("jump_penalty_base", 0.9)
        )
        _jump_gamma = (
            jump_gamma
            if jump_gamma is not None
            else defaults.get("jump_gamma", 1.0)
        )

        if _criterion not in {"entropy", "gini", "gain_ratio", "chi_square",
                                  "pig", "semantic_similarity", "pig_semantic"}:
            raise ValueError(
                f"Unsupported criterion '{_criterion}'. "
                "Use 'entropy', 'gini', 'gain_ratio', 'chi_square', "
                "'pig', 'semantic_similarity', or 'pig_semantic'."
            )

        _pig_alpha = (
            pig_alpha
            if pig_alpha is not None
            else defaults.get("pig_alpha", 1.0)
        )
        _semantic_weight = (
            semantic_weight
            if semantic_weight is not None
            else defaults.get("semantic_weight", 0.3)
        )

        # config에서 max_samples 가져오기
        if max_samples == 0:
            max_samples = ds_cfg.get("max_samples", 0)

        logger.info(
            "=== Pipeline: %s (target=%s) ===",
            dataset_name,
            target or ds_cfg["label_col"],
        )
        logger.info("Step 1/10: Loading dataset")

        # ── 1. 데이터 로드 ──
        smiles_list, labels, actual_target = load_dataset(
            dataset_name, ds_cfg, self.base_dir,
            target=target, max_samples=max_samples,
        )

        # ── 2. SMILES → 분자 디스크립터 ──
        logger.info("Step 2/10: Computing molecular descriptors")
        feature_df, valid_idx = compute_descriptors_batch(smiles_list)
        labels = labels[valid_idx]
        smiles_list = [smiles_list[i] for i in valid_idx]

        # BACE: 기존 특성 컬럼 추가
        extra_cols = ds_cfg.get("extra_feature_cols", [])
        if extra_cols:
            csv_path = self.base_dir / ds_cfg["file"]
            raw_df = pd.read_csv(str(csv_path))
            valid_mask = raw_df[actual_target].notna()
            raw_df = raw_df[valid_mask].reset_index(drop=True)
            if max_samples > 0 and len(raw_df) > max_samples:
                raw_df = raw_df.iloc[:max_samples]

            for col in extra_cols:
                if col in raw_df.columns:
                    vals = raw_df[col].iloc[valid_idx].reset_index(drop=True)
                    if pd.api.types.is_numeric_dtype(vals):
                        feature_df[col] = vals.values

        logger.info(
            "  Descriptors: %d samples, %d features",
            len(feature_df),
            len(feature_df.columns),
        )

        # ── 3. Train/Test 분할 ──
        logger.info("Step 3/10: Stratified train/test split")
        train_df, test_df, train_labels, test_labels = stratified_split(
            feature_df, labels, test_size=_test_size, seed=_seed,
        )
        logger.info(
            "  Split: train=%d (pos=%d), test=%d (pos=%d)",
            len(train_labels),
            int(np.sum(train_labels == 1)),
            len(test_labels),
            int(np.sum(test_labels == 1)),
        )

        # ── 4. 온톨로지 라우팅 & 그래프 (1:1) ──
        logger.info("Step 4/10: Routing ontology graph")
        G_base = self.router.route(
            dataset_name,
            ontology_override=ontology_override,
        )
        G = copy.deepcopy(G_base)
        bridge_domain = self.router.get_bridge_domain(
            dataset_name,
            ontology_override=ontology_override,
        )

        # ── 5. Feature → Ontology 브릿지 (domain-aware) ──
        logger.info("Step 5/10: Building feature-ontology bridge")
        bridge = SMILESOntologyBridge(G, match_threshold=65, domain=bridge_domain)
        mapping = bridge.auto_map(list(feature_df.columns))
        unmapped = [f for f in feature_df.columns if f not in mapping]
        if unmapped:
            bridge.create_feature_nodes(unmapped)

        feature_node_count = sum(
            1 for _, d in G.nodes(data=True) if d.get("is_feature")
        )
        logger.info(
            "  Bridge: %d mapped, %d created → %d feature nodes "
            "(graph: %d nodes, %d edges)",
            len(mapping),
            len(unmapped),
            feature_node_count,
            G.number_of_nodes(),
            G.number_of_edges(),
        )

        # ── 6. 도메인 지식 휴리스틱 ──
        logger.info("Step 6/10: Applying domain heuristics")
        heuristics = KnowledgeHeuristics(ds_cfg, G)
        train_df_h = heuristics.apply(train_df)
        test_df_h = heuristics.apply(test_df)

        logger.info(
            "  Heuristics: profile='%s', train_features=%d",
            heuristics.profile,
            len(train_df_h.columns),
        )

        # ── 7. 계층 게이트 설치 ──
        logger.info("Step 7/10: Installing hierarchical gates")
        hierarchical = None
        hier_cfg = ds_cfg.get("hierarchy_config")
        if hier_cfg:
            hierarchical = HierarchicalSearch(hier_cfg, G)
            n_gates = hierarchical.install_gates()
            logger.info(
                "  Hierarchy: %d gates installed, entropy=%.4f bits",
                n_gates,
                hierarchical.compute_entropy(),
            )

        # ── 결과 기본 정보 ──
        result: Dict[str, Any] = {
            "dataset": dataset_name,
            "target": actual_target,
            "n_samples": len(labels),
            "n_train": len(train_labels),
            "n_test": len(test_labels),
            "n_features": len(train_df_h.columns),
            "n_graph_nodes": G.number_of_nodes(),
            "n_graph_edges": G.number_of_edges(),
            "n_feature_nodes": feature_node_count,
            "heuristic_profile": heuristics.profile,
            "ontologies": self.router.get_required_ontologies(
                dataset_name,
                ontology_override=ontology_override,
            ),
            "criterion": _criterion,
            "jump_penalty_base": _jump_penalty_base,
            "jump_gamma": _jump_gamma,
        }

        if skip_fit:
            result["status"] = "graph_only"
            result["total_time_sec"] = time.time() - t0
            return result

        # ── 8. SemanticForest 학습 ──
        logger.info("Step 8/10: Training SemanticForest")
        forest = SemanticForest(
            graph=G,
            n_trees=_n_trees,
            n_ants_per_tree=_n_ants,
            elite_ratio=defaults.get("elite_ratio", 0.2),
            evaporation_rate=defaults.get("evaporation_rate", 0.10),
            min_pheromone=0.01,
            max_pheromone=50.0,
            bootstrap_ratio=defaults.get("bootstrap_ratio", 0.8),
            alpha=1.0,
            beta=2.0,
            max_path_length=defaults.get("max_path_length", 5),
            max_steps=defaults.get("max_steps", 80),
            min_gain=defaults.get("min_gain", 0.005),
            min_samples_leaf=max(5, int(len(train_labels) * 0.01)),
            criterion=_criterion,
            jump_penalty_base=_jump_penalty_base,
            jump_gamma=_jump_gamma,
            pig_alpha=_pig_alpha,
            semantic_weight=_semantic_weight,
            n_generations=_n_gen,
            seed=_seed,
        )

        t_fit = time.time()
        forest.fit(train_df_h, pd.Series(train_labels))
        fit_time = time.time() - t_fit

        # 고정 페로몬 복원
        if heuristics.has_static_pheromone:
            heuristics.protect_static_pheromone()

        logger.info(
            "  Training: %.2fs, %d trees, %d rules",
            fit_time,
            len(forest.trees_),
            len(forest.all_rules_),
        )

        # ── 9. 예측 & 평가 ──
        logger.info("Step 9/10: Predicting and evaluating")
        test_preds_maj = forest.predict(test_df_h, method="majority")
        test_preds_wt = forest.predict(test_df_h, method="weighted")
        test_proba = forest.predict_proba(test_df_h)

        test_m_maj = compute_metrics(test_labels, test_preds_maj, test_proba)
        test_m_wt = compute_metrics(test_labels, test_preds_wt, test_proba)

        train_preds = forest.predict(train_df_h, method="majority")
        train_proba = forest.predict_proba(train_df_h)
        train_m = compute_metrics(train_labels, train_preds, train_proba)

        # ── 10. 페로몬 분포 ──
        all_pher = [
            edata.get("pheromone", 1.0)
            for _, _, edata in G.edges(data=True)
        ]
        high_pher = sum(1 for p in all_pher if p > 1.0)

        total_time = time.time() - t0

        # ── 결과 업데이트 ──
        logger.info("Step 10/10: Aggregating result artifacts")
        result.update({
            "fit_time_sec": fit_time,
            "total_time_sec": total_time,
            "n_trees": len(forest.trees_),
            "n_rules": len(forest.all_rules_),
            "train_accuracy": train_m["accuracy"],
            "train_f1": train_m["f1"],
            "test_accuracy_majority": test_m_maj["accuracy"],
            "test_balanced_accuracy_majority": test_m_maj["balanced_accuracy"],
            "test_precision_majority": test_m_maj["precision"],
            "test_recall_majority": test_m_maj["recall"],
            "test_f1_majority": test_m_maj["f1"],
            "test_auc_roc": test_m_maj.get("auc_roc", -1),
            "test_accuracy_weighted": test_m_wt["accuracy"],
            "test_f1_weighted": test_m_wt["f1"],
            "pheromone_sparsity_pct": (
                100 * high_pher / len(all_pher) if all_pher else 0
            ),
            "feature_importance": dict(forest.get_feature_importance()),
            "hierarchy_validation": validate_rule_hierarchical_inheritance(
                G, forest.all_rules_
            ),
            "status": "complete",
        })

        hv = result["hierarchy_validation"]
        logger.info(
            "  HierarchyValidation: consistent=%d/%d (%.2f%%), hierarchy_rules=%d, contradictions=%d, order_violations=%d",
            hv["n_consistent_rules"],
            hv["n_rules_checked"],
            hv["consistency_rate"] * 100.0,
            hv["n_rules_with_hierarchy"],
            hv["n_rules_with_contradiction"],
            hv["n_rules_with_order_violation"],
        )

        logger.info(
            "  RESULT: acc=%.4f, bal_acc=%.4f, f1=%.4f, auc=%.4f (%.1fs)",
            test_m_maj["accuracy"],
            test_m_maj["balanced_accuracy"],
            test_m_maj["f1"],
            test_m_maj.get("auc_roc", -1),
            total_time,
        )

        return result

    def get_all_tasks(
        self,
        datasets: Optional[List[str]] = None,
    ) -> List[Tuple[str, str]]:
        """전체 (dataset, target) 조합 리스트를 반환한다.

        Parameters
        ----------
        datasets : list[str] | None
            실행할 데이터셋 이름 목록 (None = 전체).

        Returns
        -------
        list[tuple[str, str]]
            (dataset_name, target_name) 쌍의 리스트.
        """
        ds_names = datasets or self.router.available_datasets
        order_index = {
            name: idx for idx, name in enumerate(DATASET_EXECUTION_ORDER)
        }
        ds_names = sorted(
            ds_names,
            key=lambda name: (order_index.get(name, len(DATASET_EXECUTION_ORDER)), name),
        )
        tasks: List[Tuple[str, str]] = []
        for ds_name in ds_names:
            cfg = self.router.get_dataset_config(ds_name)
            all_targets = cfg.get("all_targets", [cfg["label_col"]])
            for tgt in all_targets:
                tasks.append((ds_name, tgt))
        return tasks

    def run_all(
        self,
        *,
        datasets: Optional[List[str]] = None,
        max_samples: int = 0,
        criterion: Optional[str] = None,
        jump_penalty_base: Optional[float] = None,
        jump_gamma: Optional[float] = None,
        pig_alpha: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        seed: int = 42,
        skip_fit: bool = False,
    ) -> List[Dict[str, Any]]:
        """데이터셋 전체 × 전체 타겟에 대해 파이프라인을 실행한다.

        Parameters
        ----------
        datasets : list[str] | None
            실행할 데이터셋 이름 목록 (None = 전체 6개).
        max_samples, criterion, jump_penalty_base, jump_gamma, pig_alpha,
        semantic_weight, seed, skip_fit : 기존과 동일.

        Returns
        -------
        list[dict]
            각 벤치마크 결과 딕셔너리 리스트.
        """
        tasks = self.get_all_tasks(datasets)
        total = len(tasks)
        logger.info("=== run_all: %d tasks across %s ===",
                     total, datasets or "all datasets")

        results = []
        for idx, (ds_name, target) in enumerate(tasks, 1):
            logger.info("── Task %d/%d: %s / %s ──", idx, total, ds_name, target)
            try:
                result = self.run(
                    ds_name,
                    target=target,
                    max_samples=max_samples,
                    criterion=criterion,
                    jump_penalty_base=jump_penalty_base,
                    jump_gamma=jump_gamma,
                    pig_alpha=pig_alpha,
                    semantic_weight=semantic_weight,
                    seed=seed,
                    skip_fit=skip_fit,
                )
                results.append(result)
            except Exception as e:
                logger.error("Failed: %s (%s): %s", ds_name, target, e)
                results.append({
                    "dataset": ds_name,
                    "target": target,
                    "status": "error",
                    "error": str(e),
                })

        return results

    def validate_graph(
        self,
        dataset_name: str,
        *,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """그래프 생성까지만 검증한다 (학습 생략).

        Returns
        -------
        dict
            검증 결과.
        """
        return self.run(
            dataset_name,
            target=target,
            skip_fit=True,
        )

    def print_summary(self) -> None:
        """파이프라인 설정 요약을 출력한다."""
        print("=" * 65)
        print(" ToxicityDataPipeline Summary")
        print("=" * 65)
        self.router.print_routing_table()
        defaults = self.router.get_defaults()
        print(f"\n  Default params: {json.dumps(defaults, indent=4)}")
        print("=" * 65)
