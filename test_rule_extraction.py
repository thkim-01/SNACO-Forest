"""
ACO Rule Extraction — 통합 테스트

Step 1: OWL → NetworkX DiGraph
Step 2: SMILES‑Ontology Bridge
Step 3: RuleExtractionEngine — 개미가 그래프를 탐색하며 의사결정 규칙 생성
"""

import sys
import os
import logging

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
from src.aco.rule_extraction import RuleExtractionEngine


def load_bbbp_features() -> tuple:
    """BBBP CSV를 로드하고 MolecularFeatureExtractor로 특성을 추출한다.

    MolecularFeatureExtractor가 없으면 합성 데이터를 사용한다.
    """
    csv_path = os.path.join(ROOT, "data", "bbbp", "BBBP.csv")
    if not os.path.exists(csv_path):
        logger.warning("BBBP.csv not found — generating synthetic data.")
        return _synthetic_data()

    df_raw = pd.read_csv(csv_path)
    logger.info("Loaded BBBP: %d rows, columns=%s", len(df_raw), list(df_raw.columns))

    # 라벨 컬럼 탐지
    label_col = None
    for candidate in ["p_np", "label", "activity", "target"]:
        if candidate in df_raw.columns:
            label_col = candidate
            break
    if label_col is None:
        # 숫자 컬럼 중 0/1 비율이 높은 것
        for col in df_raw.select_dtypes(include="number").columns:
            u = df_raw[col].dropna().unique()
            if len(u) <= 3:
                label_col = col
                break
    if label_col is None:
        logger.warning("No label column found — using synthetic data.")
        return _synthetic_data()

    labels = df_raw[label_col].dropna().astype(int)

    # SMILES → feature 추출 시도
    try:
        from src.ontology.smiles_converter import MolecularFeatureExtractor

        smiles_col = None
        for c in ["smiles", "SMILES", "mol", "Smiles"]:
            if c in df_raw.columns:
                smiles_col = c
                break
        if smiles_col is None:
            raise ValueError("No SMILES column found.")

        extractor = MolecularFeatureExtractor()
        feature_rows = []
        valid_indices = []
        for idx, smi in df_raw[smiles_col].items():
            if idx not in labels.index:
                continue
            try:
                feats = extractor.extract_features(str(smi))
                # 수치 특성만 추출
                numeric = {
                    k: v for k, v in feats.items()
                    if isinstance(v, (int, float)) and k != "_version"
                }
                feature_rows.append(numeric)
                valid_indices.append(idx)
            except Exception:
                continue

        if len(feature_rows) < 50:
            raise ValueError(f"Only {len(feature_rows)} valid molecules — too few.")

        feature_df = pd.DataFrame(feature_rows, index=valid_indices)
        labels = labels.loc[valid_indices].values

        # NaN 컬럼 제거
        feature_df = feature_df.dropna(axis=1, thresh=int(len(feature_df) * 0.8))
        feature_df = feature_df.fillna(feature_df.median())

        logger.info(
            "Extracted features: %d samples × %d features",
            len(feature_df),
            len(feature_df.columns),
        )
        return feature_df, labels

    except Exception as e:
        logger.warning("Feature extraction failed: %s — using synthetic data.", e)
        return _synthetic_data()


def _synthetic_data(n_samples: int = 500, seed: int = 42):
    """합성 데이터 생성 (rdkit 불필요)."""
    rng = np.random.RandomState(seed)
    features = {
        "molecular_weight": rng.normal(350, 100, n_samples).clip(50, 900),
        "logp": rng.normal(2.5, 1.5, n_samples),
        "tpsa": rng.normal(75, 30, n_samples).clip(0, 200),
        "num_hbd": rng.poisson(1.5, n_samples),
        "num_hba": rng.poisson(4, n_samples),
        "num_rotatable_bonds": rng.poisson(5, n_samples),
        "num_rings": rng.poisson(3, n_samples),
        "num_aromatic_rings": rng.poisson(2, n_samples),
        "num_atoms": rng.poisson(25, n_samples),
        "num_heavy_atoms": rng.poisson(20, n_samples),
        "fsp3": rng.beta(2, 5, n_samples),
    }
    df = pd.DataFrame(features)

    # 라벨: logp > 2 AND tpsa < 90 → 1 (BBB permeable) 확률 높음
    prob = 1 / (1 + np.exp(-(df["logp"] - 2) + 0.02 * (df["tpsa"] - 70)))
    labels = (rng.random(n_samples) < prob).astype(int)

    logger.info("Synthetic data: %d samples × %d features", n_samples, len(df.columns))
    return df, labels


def main():
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

    feature_df, labels = load_bbbp_features()
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
    # Step 3: Rule Extraction Engine
    # ==================================================================
    print("\n" + "=" * 72)
    print("STEP 3: ACO Rule Extraction Engine")
    print("=" * 72)

    engine = RuleExtractionEngine(
        graph=G,
        feature_df=feature_df,
        label_series=pd.Series(labels),
        alpha=1.0,
        beta=2.0,
        max_path_length=4,
        max_steps=80,
        min_gain=0.005,
        min_samples_leaf=10,
        criterion="entropy",
        seed=42,
    )

    # ── 3a: 단일 개미 규칙 추출 ──
    print("\n[3a] Single Ant Rule Extraction")
    print("-" * 40)
    single_path = engine.extract_single_rule()
    print(f"  {single_path}")
    print(f"  Rule: {single_path.to_rule_string()}")
    print(f"  Raw path length: {len(single_path.raw_path)}")
    print(f"  Feature visits:  {single_path.depth}")

    # ── 3b: 다중 개미 규칙 추출 ──
    print("\n[3b] Multi-Ant Rule Extraction (50 ants)")
    print("-" * 40)
    rules = engine.extract_rules(n_ants=50, deduplicate=True)
    print(f"  Extracted {len(rules)} unique rules")

    for i, rule in enumerate(rules[:10]):
        print(f"\n  Rule #{i + 1}:")
        print(f"    {rule.to_rule_string()}")
        print(f"    Coverage={rule.coverage}, Accuracy={rule.accuracy:.4f}, "
              f"Fitness={rule.fitness:.4f}, IG_total={rule.total_info_gain:.4f}")

    # ── 3c: 규칙 집합 평가 ──
    print("\n[3c] Rule Set Evaluation")
    print("-" * 40)
    eval_result = engine.evaluate_rules(rules)
    print(f"  Rules used:     {eval_result['n_rules']}")
    print(f"  Coverage rate:  {eval_result['coverage_rate']:.4f}")
    print(f"  Accuracy:       {eval_result['avg_accuracy']:.4f}")

    # ── 3d: 개별 샘플 예측 데모 ──
    print("\n[3d] Sample Prediction Demo")
    print("-" * 40)
    for idx in range(min(5, len(feature_df))):
        row = {col: feature_df.iloc[idx][col] for col in feature_df.columns}
        pred = None
        for rule in rules:
            pred = rule.evaluate_sample(row)
            if pred is not None:
                break
        actual = labels[idx]
        match = "✓" if pred == actual else "✗"
        print(f"  Sample {idx}: pred={pred}, actual={actual} {match}")

    print(f"\n✅ Rule Extraction Engine pipeline complete.")


if __name__ == "__main__":
    main()
