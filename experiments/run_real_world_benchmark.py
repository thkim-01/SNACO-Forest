"""
Real-World Benchmark — SemanticForest × 실제 분자 독성 데이터
=============================================================

합성 데이터를 넘어 실제 Tox21 및 ClinTox 데이터셋으로 파이프라인을 검증한다.

파이프라인:
    1. SMILES → Pure-Python 분자 기술자 (16종)
    2. DTO (Drug Target Ontology) 로딩 → NetworkX DiGraph
    3. Feature ↔ Ontology Bridge 매핑
    4. SemanticForest 앙상블 학습 (Bootstrap + Elite Pheromone)
    5. 계층화 분할(Stratified Split)로 Train/Test 분리
    6. 다수결 투표 & 가중 투표 예측 → 정확도, F1, AUC-ROC 평가
    7. TOP-5 페로몬 경로 해석 + Feature Importance 보고

사용법:
    python experiments/run_real_world_benchmark.py --dataset tox21 --target NR-AhR
    python experiments/run_real_world_benchmark.py --dataset clintox --target CT_TOX
    python experiments/run_real_world_benchmark.py --all
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Windows cp949 인코딩 문제 해결
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd

# ── 경로 설정 ──
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.aco.owl_graph_builder import OWLGraphBuilder
from src.aco.smiles_ontology_bridge import SMILESOntologyBridge
from src.aco.smiles_descriptor import compute_descriptors_batch
from src.aco.semantic_forest import SemanticForest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("benchmark")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 로더
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_tox21(target: str = "NR-AhR") -> Tuple[List[str], np.ndarray]:
    """Tox21 데이터셋의 특정 타겟을 로드한다.

    Parameters
    ----------
    target : str
        독성 어세이 타겟명. 예: 'NR-AhR', 'SR-MMP', 'NR-AR', ...

    Returns
    -------
    (smiles_list, labels)
    """
    csv_path = os.path.join(ROOT_DIR, "data", "tox21", "tox21.csv")
    df = pd.read_csv(csv_path)

    if target not in df.columns:
        available = [c for c in df.columns if c not in ("mol_id", "smiles")]
        raise ValueError(
            f"Target '{target}' not found in Tox21. Available: {available}"
        )

    # NaN 라벨 제거
    valid = df[target].notna()
    smiles = df.loc[valid, "smiles"].tolist()
    labels = df.loc[valid, target].astype(int).values

    logger.info(
        "Tox21 [%s]: %d samples (pos=%d, neg=%d, ratio=%.3f)",
        target,
        len(labels),
        (labels == 1).sum(),
        (labels == 0).sum(),
        (labels == 1).mean(),
    )
    return smiles, labels


def load_clintox(target: str = "CT_TOX") -> Tuple[List[str], np.ndarray]:
    """ClinTox 데이터셋을 로드한다.

    Parameters
    ----------
    target : str
        'CT_TOX' 또는 'FDA_APPROVED'.

    Returns
    -------
    (smiles_list, labels)
    """
    csv_path = os.path.join(ROOT_DIR, "data", "clintox", "clintox.csv")
    df = pd.read_csv(csv_path)

    smiles = df["smiles"].tolist()
    labels = df[target].astype(int).values

    logger.info(
        "ClinTox [%s]: %d samples (pos=%d, neg=%d, ratio=%.3f)",
        target,
        len(labels),
        (labels == 1).sum(),
        (labels == 0).sum(),
        (labels == 1).mean(),
    )
    return smiles, labels


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Train/Test 분할 (계층화)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def stratified_split(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """계층화 랜덤 분할.

    각 클래스의 비율을 유지하면서 train/test로 분리한다.

    Returns
    -------
    (train_df, test_df, train_labels, test_labels)
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    n = len(labels)

    # 클래스별 인덱스
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 평가 메트릭
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """정확도, F1, Precision, Recall, AUC-ROC를 계산한다."""
    n = len(y_true)
    if n == 0:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    # 정확도
    accuracy = float((y_true == y_pred).sum() / n)

    # 이진 분류 메트릭
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # 균형 정확도 (Balanced Accuracy)
    sensitivity = recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2

    metrics = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    # AUC-ROC (NumPy 순수 구현)
    if y_proba is not None and y_proba.shape[1] >= 2:
        pos_proba = y_proba[:, 1]
        metrics["auc_roc"] = _auc_roc(y_true, pos_proba)

    return metrics


def _auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """순수 NumPy AUC-ROC 계산 (trapezoidal rule)."""
    # 점수 내림차순 정렬
    desc_idx = np.argsort(-y_scores)
    y_sorted = y_true[desc_idx]

    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp_acc = 0
    fp_acc = 0

    for label in y_sorted:
        if label == 1:
            tp_acc += 1
        else:
            fp_acc += 1
        tpr_list.append(tp_acc / n_pos)
        fpr_list.append(fp_acc / n_neg)

    # 사다리꼴 적분
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return auc


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 온톨로지 로딩 & 그래프 준비
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_ontology_graph(
    owl_filename: str = "DTO.xrdf",
) -> "nx.DiGraph":
    """온톨로지 파일을 로드하여 NetworkX DiGraph로 변환한다."""
    import networkx as nx

    owl_path = os.path.join(ROOT_DIR, "ontology", owl_filename)
    if not os.path.exists(owl_path):
        raise FileNotFoundError(f"Ontology file not found: {owl_path}")

    logger.info("Loading ontology: %s", owl_path)
    builder = OWLGraphBuilder(owl_path, default_pheromone=1.0)
    G = builder.build()
    logger.info(
        "Ontology graph: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 단일 벤치마크 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_single_benchmark(
    dataset_name: str,
    target: str,
    G: "nx.DiGraph",
    *,
    n_trees: int = 8,
    n_ants_per_tree: int = 25,
    n_generations: int = 3,
    elite_ratio: float = 0.2,
    evaporation_rate: float = 0.10,
    test_size: float = 0.2,
    seed: int = 42,
    max_samples: int = 0,
    bridge_domain: str = "anchor",
) -> Dict[str, Any]:
    """단일 데이터셋/타겟에 대해 전체 벤치마크를 실행한다.

    Parameters
    ----------
    dataset_name : str
        'tox21' 또는 'clintox'.
    target : str
        타겟 라벨 컬럼명.
    G : nx.DiGraph
        사전 로드된 온톨로지 그래프.
    max_samples : int
        0이면 전체, >0 이면 최대 샘플 수 제한 (빠른 테스트용).

    Returns
    -------
    dict
        벤치마크 결과.
    """
    import copy

    t0 = time.time()

    header = f"{'=' * 72}\n  BENCHMARK: {dataset_name.upper()} - {target}\n{'=' * 72}"
    print(f"\n{header}")

    # ── 1. 데이터 로드 ──
    print("\n[1] Load Dataset")
    print("-" * 40)

    if dataset_name == "tox21":
        smiles_list, labels = load_tox21(target)
    elif dataset_name == "clintox":
        smiles_list, labels = load_clintox(target)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 샘플 수 제한 (선택적)
    if max_samples > 0 and len(smiles_list) > max_samples:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(smiles_list), max_samples, replace=False)
        smiles_list = [smiles_list[i] for i in idx]
        labels = labels[idx]
        logger.info("Subsampled to %d samples.", max_samples)

    print(f"  Samples: {len(labels)}")
    print(f"  Positive rate: {(labels == 1).mean():.3f}")

    # ── 2. SMILES → 분자 기술자 ──
    print("\n[2] SMILES → Molecular Descriptors (Pure Python)")
    print("-" * 40)

    t_desc = time.time()
    feature_df, valid_indices = compute_descriptors_batch(
        smiles_list, min_valid_ratio=0.3, verbose=True
    )
    labels = labels[valid_indices]
    desc_time = time.time() - t_desc

    print(f"  Valid molecules: {len(feature_df)}/{len(smiles_list)}")
    print(f"  Features: {list(feature_df.columns)}")
    print(f"  Descriptor time: {desc_time:.2f}s")

    # NaN 처리
    feature_df = feature_df.fillna(feature_df.median())

    # ── 3. Train/Test 분할 ──
    print("\n[3] Stratified Train/Test Split")
    print("-" * 40)

    train_df, test_df, train_labels, test_labels = stratified_split(
        feature_df, labels, test_size=test_size, seed=seed
    )

    print(f"  Train: {len(train_labels)} (pos={sum(train_labels==1)}, neg={sum(train_labels==0)})")
    print(f"  Test:  {len(test_labels)} (pos={sum(test_labels==1)}, neg={sum(test_labels==0)})")

    # ── 4. Feature ↔ Ontology Bridge ──
    print("\n[4] Feature ↔ Ontology Bridge")
    print("-" * 40)

    # 각 벤치마크에서 그래프의 페로몬을 초기화 (이전 실행의 영향 제거)
    G_fresh = copy.deepcopy(G)

    bridge = SMILESOntologyBridge(G_fresh, match_threshold=65, domain=bridge_domain)
    mapping = bridge.auto_map(list(feature_df.columns))

    unmapped = [f for f in feature_df.columns if f not in mapping]
    if unmapped:
        bridge.create_feature_nodes(unmapped)

    bridge.print_mapping()

    feature_node_count = sum(
        1 for _, d in G_fresh.nodes(data=True) if d.get("is_feature")
    )
    print(f"  Feature nodes in ontology: {feature_node_count}")
    print(f"  Graph: {G_fresh.number_of_nodes()} nodes, {G_fresh.number_of_edges()} edges")

    # ── 5. SemanticForest 학습 ──
    print("\n[5] SemanticForest Training")
    print("-" * 40)

    forest = SemanticForest(
        graph=G_fresh,
        n_trees=n_trees,
        n_ants_per_tree=n_ants_per_tree,
        elite_ratio=elite_ratio,
        evaporation_rate=evaporation_rate,
        min_pheromone=0.01,
        max_pheromone=50.0,
        bootstrap_ratio=0.8,
        alpha=1.0,
        beta=2.0,
        max_path_length=5,
        max_steps=80,
        min_gain=0.005,
        min_samples_leaf=max(5, int(len(train_labels) * 0.01)),
        criterion="entropy",
        n_generations=n_generations,
        seed=seed,
    )

    t_fit = time.time()
    forest.fit(train_df, pd.Series(train_labels))
    fit_time = time.time() - t_fit

    print(f"  Training time: {fit_time:.2f}s")
    print(f"  Trees: {len(forest.trees_)}, Unique rules: {len(forest.all_rules_)}")

    # ── 6. 예측 & 평가 ──
    print("\n[6] Prediction & Evaluation")
    print("-" * 40)

    # Train 성능
    train_preds_maj = forest.predict(train_df, method="majority")
    train_proba = forest.predict_proba(train_df)
    train_metrics_maj = compute_metrics(train_labels, train_preds_maj, train_proba)

    # Test 성능 (핵심!)
    test_preds_maj = forest.predict(test_df, method="majority")
    test_preds_wt = forest.predict(test_df, method="weighted")
    test_proba = forest.predict_proba(test_df)
    test_metrics_maj = compute_metrics(test_labels, test_preds_maj, test_proba)
    test_metrics_wt = compute_metrics(test_labels, test_preds_wt, test_proba)

    print("\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  TRAIN (Majority)                                   │")
    print(f"  │    Accuracy:          {train_metrics_maj['accuracy']:.4f}                        │")
    print(f"  │    Balanced Accuracy: {train_metrics_maj['balanced_accuracy']:.4f}                        │")
    print(f"  │    F1 Score:          {train_metrics_maj['f1']:.4f}                        │")
    auc_train = train_metrics_maj.get('auc_roc', -1)
    if auc_train >= 0:
        print(f"  │    AUC-ROC:           {auc_train:.4f}                        │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  TEST — Majority Voting                             │")
    print(f"  │    Accuracy:          {test_metrics_maj['accuracy']:.4f}                        │")
    print(f"  │    Balanced Accuracy: {test_metrics_maj['balanced_accuracy']:.4f}                        │")
    print(f"  │    Precision:         {test_metrics_maj['precision']:.4f}                        │")
    print(f"  │    Recall:            {test_metrics_maj['recall']:.4f}                        │")
    print(f"  │    F1 Score:          {test_metrics_maj['f1']:.4f}                        │")
    auc_test = test_metrics_maj.get('auc_roc', -1)
    if auc_test >= 0:
        print(f"  │    AUC-ROC:           {auc_test:.4f}                        │")
    print(f"  ├─────────────────────────────────────────────────────┤")
    print(f"  │  TEST — Weighted Voting                             │")
    print(f"  │    Accuracy:          {test_metrics_wt['accuracy']:.4f}                        │")
    print(f"  │    Balanced Accuracy: {test_metrics_wt['balanced_accuracy']:.4f}                        │")
    print(f"  │    F1 Score:          {test_metrics_wt['f1']:.4f}                        │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # ── 7. TOP-5 페로몬 경로 해석 ──
    print("\n[7] TOP-5 Pheromone Path Interpretation")
    print("-" * 40)

    top_paths = forest.interpret(top_k=5)
    for i, p in enumerate(top_paths):
        print(f"\n  * TOP-{i + 1}:")
        print(f"    Rule:          {p.rule_string}")
        print(f"    Avg Pheromone: {p.avg_pheromone:.4f}")
        print(f"    Fitness:       {p.fitness:.4f}")
        print(f"    Accuracy:      {p.accuracy:.4f} (coverage={p.coverage})")
        print(f"    Ontology path: {' -> '.join(p.ontology_path[:6])}"
              + (" ..." if len(p.ontology_path) > 6 else ""))

    # ── 8. Feature Importance ──
    print("\n[8] Feature Importance")
    print("-" * 40)

    importance = forest.get_feature_importance()
    max_imp = max(importance.values()) if importance else 1.0
    for feat, score in importance.items():
        bar_len = int(score / max_imp * 40) if max_imp > 0 else 0
        bar = "#" * bar_len
        print(f"  {feat:25s} {score:.6f} {bar}")

    # ── 9. 페로몬 분포 통계 ──
    print("\n[9] Pheromone Distribution")
    print("-" * 40)

    all_pher = [
        edata.get("pheromone", 1.0)
        for _, _, edata in G_fresh.edges(data=True)
    ]
    high_pher = sum(1 for p in all_pher if p > 1.0)
    print(f"  Total edges:  {len(all_pher)}")
    print(f"  Mean:         {np.mean(all_pher):.4f}")
    print(f"  Std:          {np.std(all_pher):.4f}")
    print(f"  Min/Max:      {np.min(all_pher):.4f} / {np.max(all_pher):.4f}")
    print(f"  Sparsity:     {high_pher}/{len(all_pher)} edges > 1.0 "
          f"({100 * high_pher / len(all_pher):.2f}%)")

    total_time = time.time() - t0

    # ── 결과 수집 ──
    result = {
        "dataset": dataset_name,
        "target": target,
        "n_samples": len(labels),
        "n_train": len(train_labels),
        "n_test": len(test_labels),
        "n_features": len(feature_df.columns),
        "n_trees": len(forest.trees_),
        "n_rules": len(forest.all_rules_),
        "train_accuracy": train_metrics_maj["accuracy"],
        "train_f1": train_metrics_maj["f1"],
        "test_accuracy_majority": test_metrics_maj["accuracy"],
        "test_balanced_accuracy_majority": test_metrics_maj["balanced_accuracy"],
        "test_f1_majority": test_metrics_maj["f1"],
        "test_auc_roc": test_metrics_maj.get("auc_roc", -1),
        "test_accuracy_weighted": test_metrics_wt["accuracy"],
        "test_f1_weighted": test_metrics_wt["f1"],
        "pheromone_sparsity_pct": 100 * high_pher / len(all_pher),
        "top1_rule": top_paths[0].rule_string if top_paths else "",
        "top1_pheromone": top_paths[0].avg_pheromone if top_paths else 0.0,
        "fit_time_sec": fit_time,
        "total_time_sec": total_time,
        "feature_importance": importance,
    }

    print(f"\n  >> Total benchmark time: {total_time:.2f}s")
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 전체 벤치마크 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_all_benchmarks(
    *,
    max_samples: int = 0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """모든 데이터셋/타겟에 대해 벤치마크를 실행한다."""

    # 1:1 routing via OntologyRouter
    from src.aco.ontology_router import OntologyRouter
    router = OntologyRouter()

    benchmarks = [
        # Tox21 — 주요 독성 타겟 (양성 비율이 적절한 것)
        ("tox21", "NR-AhR"),
        ("tox21", "SR-MMP"),
        ("tox21", "SR-ARE"),
        ("tox21", "SR-p53"),
        # ClinTox
        ("clintox", "CT_TOX"),
        ("clintox", "FDA_APPROVED"),
    ]

    results = []
    for dataset_name, target in benchmarks:
        try:
            G = router.route(dataset_name)
            bd = router.get_bridge_domain(dataset_name)
            result = run_single_benchmark(
                dataset_name,
                target,
                G,
                n_trees=8,
                n_ants_per_tree=25,
                n_generations=3,
                elite_ratio=0.2,
                evaporation_rate=0.10,
                test_size=0.2,
                seed=seed,
                max_samples=max_samples,
                bridge_domain=bd,
            )
            results.append(result)
        except Exception as e:
            logger.error("Benchmark failed for %s/%s: %s", dataset_name, target, e)
            import traceback
            traceback.print_exc()

    # ── 종합 결과 테이블 ──
    print("\n" + "=" * 88)
    print("  BENCHMARK SUMMARY")
    print("=" * 88)
    print(f"  {'Dataset':<10s} {'Target':<15s} {'N':<6s} "
          f"{'TestAcc':<10s} {'TestBAcc':<10s} {'TestF1':<10s} "
          f"{'AUC-ROC':<10s} {'Sparsity':<10s}")
    print("-" * 88)

    for r in results:
        auc = f"{r['test_auc_roc']:.4f}" if r['test_auc_roc'] >= 0 else "N/A"
        print(
            f"  {r['dataset']:<10s} {r['target']:<15s} {r['n_samples']:<6d} "
            f"{r['test_accuracy_majority']:<10.4f} "
            f"{r['test_balanced_accuracy_majority']:<10.4f} "
            f"{r['test_f1_majority']:<10.4f} "
            f"{auc:<10s} "
            f"{r['pheromone_sparsity_pct']:<10.2f}%"
        )

    print("=" * 88)

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main():
    parser = argparse.ArgumentParser(
        description="SemanticForest Real-World Benchmark"
    )
    parser.add_argument(
        "--dataset",
        choices=["tox21", "clintox", "all"],
        default="all",
        help="Dataset to benchmark.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target label column (e.g., NR-AhR, CT_TOX).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples per dataset (0=all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    if args.dataset == "all":
        run_all_benchmarks(max_samples=args.max_samples, seed=args.seed)
    else:
        from src.aco.ontology_router import OntologyRouter
        router = OntologyRouter()
        G = router.route(args.dataset)
        bd = router.get_bridge_domain(args.dataset)

        target = args.target
        if target is None:
            target = "NR-AhR" if args.dataset == "tox21" else "CT_TOX"

        run_single_benchmark(
            args.dataset,
            target,
            G,
            n_trees=8,
            n_ants_per_tree=25,
            n_generations=3,
            elite_ratio=0.2,
            evaporation_rate=0.10,
            test_size=0.2,
            seed=args.seed,
            max_samples=args.max_samples,
            bridge_domain=bd,
        )


if __name__ == "__main__":
    main()
