"""
SemanticForest — 통합 테스트

Step 1: OWL → NetworkX DiGraph
Step 2: SMILES Feature ↔ Ontology Bridge
Step 3: SemanticForest 앙상블 학습 + 예측 + 해석
"""

import sys
import os
import logging
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from src.aco.owl_graph_builder import OWLGraphBuilder
from src.aco.smiles_ontology_bridge import SMILESOntologyBridge
from src.aco.semantic_forest import SemanticForest


# ── 합성 데이터 생성 ──────────────────────────────────────────


def _synthetic_data(n_samples: int = 500, seed: int = 42):
    """합성 데이터 생성 (rdkit 불필요)."""
    rng = np.random.RandomState(seed)
    features = {
        "molecular_weight": rng.normal(350, 100, n_samples).clip(50, 900),
        "logp": rng.normal(2.5, 1.5, n_samples),
        "tpsa": rng.normal(75, 30, n_samples).clip(0, 200),
        "num_hbd": rng.poisson(1.5, n_samples).astype(float),
        "num_hba": rng.poisson(4, n_samples).astype(float),
        "num_rotatable_bonds": rng.poisson(5, n_samples).astype(float),
        "num_rings": rng.poisson(3, n_samples).astype(float),
        "num_aromatic_rings": rng.poisson(2, n_samples).astype(float),
        "num_atoms": rng.poisson(25, n_samples).astype(float),
        "num_heavy_atoms": rng.poisson(20, n_samples).astype(float),
        "fsp3": rng.beta(2, 5, n_samples),
    }
    df = pd.DataFrame(features)

    # 라벨: logp > 2 AND tpsa < 90 → 1 (BBB permeable) 확률 높음
    prob = 1 / (1 + np.exp(-(df["logp"] - 2) + 0.02 * (df["tpsa"] - 70)))
    labels = (rng.random(n_samples) < prob).astype(int)

    logger.info("Synthetic data: %d samples × %d features", n_samples, len(df.columns))
    return df, labels


def main():
    t0 = time.time()

    # ==================================================================
    # Step 1: OWL → NetworkX DiGraph
    # ==================================================================
    print("=" * 72)
    print("STEP 1: OWL → NetworkX DiGraph")
    print("=" * 72)

    owl_path = os.path.join(ROOT, "ontology", "DTO.xrdf")
    if not os.path.exists(owl_path):
        owl_path = os.path.join(ROOT, "ontology", "DTO.owl")

    builder = OWLGraphBuilder(owl_path, default_pheromone=1.0)
    G = builder.build()
    builder.print_summary()

    # ==================================================================
    # Step 2: Feature Bridge
    # ==================================================================
    print("\n" + "=" * 72)
    print("STEP 2: SMILES Feature ↔ Ontology Bridge")
    print("=" * 72)

    feature_df, labels = _synthetic_data(n_samples=500, seed=42)
    feature_names = list(feature_df.columns)

    bridge = SMILESOntologyBridge(G, match_threshold=65)
    mapping = bridge.auto_map(feature_names)

    unmapped = [f for f in feature_names if f not in mapping]
    if unmapped:
        bridge.create_feature_nodes(unmapped)

    bridge.print_mapping()

    feature_node_count = sum(
        1 for _, d in G.nodes(data=True) if d.get("is_feature")
    )
    print(f"\nTotal is_feature nodes: {feature_node_count}")
    print(f"Dataset: {len(labels)} samples, {len(feature_names)} features")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # ==================================================================
    # Step 3: SemanticForest Ensemble
    # ==================================================================
    print("\n" + "=" * 72)
    print("STEP 3: SemanticForest Ensemble")
    print("=" * 72)

    # ── 3a: 초기화 ──
    print("\n[3a] Initialize SemanticForest")
    print("-" * 40)

    forest = SemanticForest(
        graph=G,
        n_trees=5,
        n_ants_per_tree=20,
        elite_ratio=0.2,
        evaporation_rate=0.1,
        min_pheromone=0.01,
        max_pheromone=50.0,
        bootstrap_ratio=0.8,
        alpha=1.0,
        beta=2.0,
        max_path_length=4,
        max_steps=80,
        min_gain=0.005,
        min_samples_leaf=10,
        criterion="entropy",
        n_generations=2,
        seed=42,
    )
    print(f"  {forest}")

    # ── 3b: 학습 ──
    print("\n[3b] Fit (Bootstrap Bagging + Elite Pheromone)")
    print("-" * 40)

    t_fit = time.time()
    forest.fit(feature_df, pd.Series(labels))
    fit_time = time.time() - t_fit

    print(f"  Fit time: {fit_time:.2f}s")
    print(f"  Trees: {len(forest.trees_)}")
    print(f"  Total unique rules: {len(forest.all_rules_)}")
    print(f"  {forest}")

    # ── 3c: 예측 (다수결 투표) ──
    print("\n[3c] Predict — Majority Voting")
    print("-" * 40)

    preds_majority = forest.predict(feature_df, method="majority")
    acc_majority = (preds_majority == labels).mean()
    print(f"  Majority voting accuracy (train): {acc_majority:.4f}")
    print(f"  Prediction distribution: {dict(zip(*np.unique(preds_majority, return_counts=True)))}")

    # ── 3d: 예측 (가중 투표) ──
    print("\n[3d] Predict — Weighted Voting")
    print("-" * 40)

    preds_weighted = forest.predict(feature_df, method="weighted")
    acc_weighted = (preds_weighted == labels).mean()
    print(f"  Weighted voting accuracy (train): {acc_weighted:.4f}")
    print(f"  Prediction distribution: {dict(zip(*np.unique(preds_weighted, return_counts=True)))}")

    # ── 3e: 확률 추정 ──
    print("\n[3e] Predict Probabilities")
    print("-" * 40)

    proba = forest.predict_proba(feature_df)
    print(f"  Proba shape: {proba.shape}")
    print(f"  Sample 0 proba: {proba[0]}")
    print(f"  Sample 1 proba: {proba[1]}")
    print(f"  Mean proba (class 0): {proba[:, 0].mean():.4f}")
    if proba.shape[1] > 1:
        print(f"  Mean proba (class 1): {proba[:, 1].mean():.4f}")

    # ── 3f: 해석 — TOP-5 페로몬 경로 ──
    print("\n[3f] Interpret — TOP-5 Pheromone Paths")
    print("-" * 40)

    top_paths = forest.interpret(top_k=5)
    for i, path_info in enumerate(top_paths):
        print(f"\n  TOP-{i + 1}:")
        print(f"    Rule: {path_info.rule_string}")
        print(f"    Avg Pheromone: {path_info.avg_pheromone:.4f}")
        print(f"    Max Pheromone: {path_info.max_pheromone:.4f}")
        print(f"    Fitness: {path_info.fitness:.4f}")
        print(f"    Accuracy: {path_info.accuracy:.4f}")
        print(f"    Coverage: {path_info.coverage}")
        print(f"    Tree: #{path_info.tree_index}")
        print(f"    Path length: {len(path_info.ontology_path)} nodes")

    # ── 3g: Feature Importance ──
    print("\n[3g] Feature Importance")
    print("-" * 40)

    importance = forest.get_feature_importance()
    for feat, score in importance.items():
        bar = "█" * int(score * 50 / max(importance.values(), default=1))
        print(f"  {feat:25s} {score:.6f} {bar}")

    # ── 3h: 전체 요약 ──
    print("\n[3h] Summary")
    print("-" * 40)

    summary = forest.summary()
    for key, val in summary.items():
        if key == "feature_importance":
            continue
        print(f"  {key}: {val}")

    # ── 3i: 페로몬 분포 확인 ──
    print("\n[3i] Pheromone Distribution After Training")
    print("-" * 40)

    all_pher = [
        edata.get("pheromone", 1.0)
        for _, _, edata in G.edges(data=True)
    ]
    print(f"  Total edges: {len(all_pher)}")
    print(f"  Mean pheromone:   {np.mean(all_pher):.4f}")
    print(f"  Std pheromone:    {np.std(all_pher):.4f}")
    print(f"  Min pheromone:    {np.min(all_pher):.4f}")
    print(f"  Max pheromone:    {np.max(all_pher):.4f}")
    print(f"  Median pheromone: {np.median(all_pher):.4f}")

    # 페로몬 > 초기값(1.0) 인 엣지 비율
    high_pher = sum(1 for p in all_pher if p > 1.0)
    print(f"  Edges with pheromone > 1.0: {high_pher}/{len(all_pher)} "
          f"({100 * high_pher / len(all_pher):.1f}%)")

    total_time = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"✅ SemanticForest pipeline complete. Total time: {total_time:.2f}s")
    print(f"   Ensemble accuracy (majority): {acc_majority:.4f}")
    print(f"   Ensemble accuracy (weighted): {acc_weighted:.4f}")
    print(f"   TOP-1 path pheromone: {top_paths[0].avg_pheromone:.4f}" if top_paths else "   No paths found")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
