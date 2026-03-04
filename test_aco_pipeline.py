"""
ACO Semantic Forest — 통합 테스트 스크립트

Step 1: OWL → NetworkX DiGraph (pheromone 엣지)
Step 2: SMILES 특성 ↔ 온톨로지 노드 브릿지
Step 3: 개미 확률적 이동 시뮬레이션
"""

import sys
import os
import logging

# 프로젝트 루트를 sys.path에 추가
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from src.aco.owl_graph_builder import OWLGraphBuilder
from src.aco.smiles_ontology_bridge import SMILESOntologyBridge
from src.aco.ant_path_finder import AntPathFinder


def main():
    # ================================================================
    # Step 1: OWL → NetworkX DiGraph
    # ================================================================
    print("=" * 70)
    print("STEP 1: OWL → NetworkX DiGraph")
    print("=" * 70)

    # DTO.xrdf는 286K 트리플을 포함한 가장 풍부한 OWL 파일
    owl_path = os.path.join(ROOT, "ontology", "DTO.xrdf")
    if not os.path.exists(owl_path):
        owl_path = os.path.join(ROOT, "ontology", "DTO.owl")

    builder = OWLGraphBuilder(owl_path, default_pheromone=1.0)
    G = builder.build()
    builder.print_summary()

    # 노드 샘플 조회
    sample_nodes = list(G.nodes)[:5]
    print(f"\nSample nodes: {sample_nodes}")
    if sample_nodes:
        nbrs = builder.get_neighbors(sample_nodes[0])
        print(f"\nNeighbors of '{sample_nodes[0]}' ({len(nbrs)} total):")
        for n in nbrs[:5]:
            print(f"  → {n['label']} (pred={n['predicate']}, "
                  f"pheromone={n['pheromone']:.2f}, dir={n['direction']})")

    # ================================================================
    # Step 2: SMILES Feature ↔ Ontology Bridge
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: SMILES Feature ↔ Ontology Bridge")
    print("=" * 70)

    bridge = SMILESOntologyBridge(G, match_threshold=65)

    feature_names = [
        "molecular_weight", "logp", "tpsa",
        "num_hbd", "num_hba", "num_rotatable_bonds",
        "num_rings", "num_aromatic_rings",
        "num_atoms", "num_heavy_atoms", "fsp3",
    ]

    mapping = bridge.auto_map(feature_names)
    print(f"\nAuto-mapped: {len(mapping)} features")

    # 매핑 안 된 특성은 신규 노드로 생성
    unmapped = [f for f in feature_names if f not in mapping]
    if unmapped:
        print(f"Creating new feature nodes for {len(unmapped)} unmapped features...")
        bridge.create_feature_nodes(unmapped)

    bridge.print_mapping()

    # is_feature 노드 확인
    feature_node_count = sum(
        1 for _, d in G.nodes(data=True) if d.get("is_feature")
    )
    print(f"\nTotal is_feature nodes in graph: {feature_node_count}")

    # ================================================================
    # Step 3: Ant Path Simulation
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Ant Path Simulation")
    print("=" * 70)

    ant = AntPathFinder(
        G,
        alpha=1.0,
        beta=2.0,
        feature_heuristic=3.0,
        default_heuristic=1.0,
        max_steps=30,
        seed=42,
    )

    # 단일 개미 탐색 — "Thing" (hub) 에서 시작하여 feature 노드 도달 확인
    start = "Thing" if "Thing" in G else None
    path = ant.explore(start_node=start)
    summary = ant.path_summary(path)
    print(f"\n[Single Ant]")
    print(f"  Path length      : {summary['length']}")
    print(f"  Unique nodes     : {summary['unique_nodes']}")
    print(f"  Feature visits   : {summary['feature_count']}")
    print(f"  Total pheromone  : {summary['total_pheromone']}")
    print(f"\n  Path trajectory:")
    for i, node in enumerate(path):
        marker = " ★ FEATURE" if G.nodes[node].get("is_feature") else ""
        print(f"    [{i:2d}] {node}{marker}")

    # 페로몬 업데이트
    print("\n  Updating pheromones (deposit=2.0, evaporation=0.05)...")
    ant.update_pheromones(path, deposit=2.0, evaporation=0.05)

    # 업데이트 후 재탐색
    path2 = ant.explore(start_node=path[0])
    summary2 = ant.path_summary(path2)
    print(f"\n[After Pheromone Update — Same start]")
    print(f"  Path length      : {summary2['length']}")
    print(f"  Feature visits   : {summary2['feature_count']}")
    print(f"  Total pheromone  : {summary2['total_pheromone']}")

    # 다중 개미 탐색
    print("\n[Multi-Ant Exploration — 5 ants]")
    paths = ant.explore_multi(n_ants=5)
    for idx, p in enumerate(paths):
        s = ant.path_summary(p)
        print(f"  Ant {idx}: len={s['length']}, features={s['feature_count']}, "
              f"pheromone={s['total_pheromone']}")

    print("\n✅ All 3 steps completed successfully.")


if __name__ == "__main__":
    main()
