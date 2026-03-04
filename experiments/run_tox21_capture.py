"""Minimal runner to capture Tox21 NR-AhR benchmark results to file."""
import sys
import os
import json
import time
import traceback

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

LOG = "output/tox21_run_log.txt"

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + "\n")

try:
    # Clear log
    with open(LOG, 'w', encoding='utf-8') as f:
        f.write("")

    log("Starting Tox21 NR-AhR benchmark")

    log("Step 1: Importing modules...")
    from experiments.run_real_world_benchmark import (
        load_ontology_graph, load_tox21, stratified_split, compute_metrics
    )
    from src.aco import (
        SMILESOntologyBridge, SemanticForest, compute_descriptors_batch
    )
    from src.aco.ontology_router import OntologyRouter
    import copy
    import numpy as np
    import pandas as pd

    log("Step 2: Loading ontology graph (GO via OntologyRouter)...")
    router = OntologyRouter()
    G = router.route("tox21")
    bridge_domain = router.get_bridge_domain("tox21")
    log(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, domain={bridge_domain}")

    log("Step 3: Loading dataset...")
    smiles_list, labels = load_tox21('NR-AhR')
    log(f"  Samples: {len(labels)}, Positive: {sum(labels)}, Negative: {len(labels)-sum(labels)}")

    log("Step 4: Computing descriptors...")
    feature_df, valid_idx = compute_descriptors_batch(smiles_list)
    labels = [labels[i] for i in valid_idx]
    smiles_list = [smiles_list[i] for i in valid_idx]
    log(f"  Valid: {len(labels)}, Features: {len(feature_df.columns)}")

    log("Step 5: Splitting data...")
    labels = np.array(labels)
    (train_df, test_df, train_labels, test_labels) = stratified_split(feature_df, labels, test_size=0.2, seed=42)
    log(f"  Train: {len(train_labels)} (pos={sum(train_labels)}), Test: {len(test_labels)} (pos={sum(test_labels)})")

    log("Step 6: Setting up ontology bridge...")
    G_fresh = copy.deepcopy(G)
    bridge = SMILESOntologyBridge(G_fresh, match_threshold=65, domain=bridge_domain)
    feature_cols = list(train_df.columns)
    mapping = bridge.auto_map(feature_cols)
    unmapped = [f for f in feature_cols if f not in mapping]
    if unmapped:
        bridge.create_feature_nodes(unmapped)
    feature_node_count = sum(1 for _, d in G_fresh.nodes(data=True) if d.get("is_feature"))
    log(f"  Mapped {len(mapping)} features, created {len(unmapped)} feature nodes, total feature nodes: {feature_node_count}")
    log(f"  Graph: {G_fresh.number_of_nodes()} nodes, {G_fresh.number_of_edges()} edges")

    log("Step 7: Training SemanticForest...")
    forest = SemanticForest(
        graph=G_fresh,
        n_trees=8,
        n_ants_per_tree=25,
        elite_ratio=0.2,
        evaporation_rate=0.10,
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
        n_generations=3,
        seed=42,
    )
    t0 = time.time()
    forest.fit(train_df, pd.Series(train_labels))
    fit_time = time.time() - t0
    log(f"  Training done in {fit_time:.2f}s")
    log(f"  Trees: {len(forest.trees_)}, Rules: {len(forest.all_rules_)}")

    log("Step 8: Predicting...")
    train_preds = forest.predict(train_df, method="majority")
    test_preds_maj = forest.predict(test_df, method="majority")
    test_preds_wt = forest.predict(test_df, method="weighted")
    test_proba = forest.predict_proba(test_df)
    train_proba = forest.predict_proba(train_df)
    log("  Predictions done")

    log("Step 9: Computing metrics...")
    train_m = compute_metrics(train_labels, train_preds, train_proba)
    test_m_maj = compute_metrics(test_labels, test_preds_maj, test_proba)
    test_m_wt = compute_metrics(test_labels, test_preds_wt, test_proba)

    log("===== RESULTS =====")
    log(f"TRAIN  Accuracy={train_m['accuracy']:.4f}  BalAcc={train_m['balanced_accuracy']:.4f}  F1={train_m['f1']:.4f}")
    log(f"TEST(Majority)  Accuracy={test_m_maj['accuracy']:.4f}  BalAcc={test_m_maj['balanced_accuracy']:.4f}  "
        f"Precision={test_m_maj['precision']:.4f}  Recall={test_m_maj['recall']:.4f}  F1={test_m_maj['f1']:.4f}  "
        f"AUC-ROC={test_m_maj.get('auc_roc', -1):.4f}")
    log(f"TEST(Weighted)  Accuracy={test_m_wt['accuracy']:.4f}  BalAcc={test_m_wt['balanced_accuracy']:.4f}  F1={test_m_wt['f1']:.4f}")

    log("Step 10: Interpreting top paths...")
    top_paths = forest.interpret(top_k=5)
    for i, p in enumerate(top_paths):
        log(f"  TOP-{i+1}: {p.rule_string}  pheromone={p.avg_pheromone:.4f}  acc={p.accuracy:.4f}  cov={p.coverage}")

    log("Step 11: Feature importance...")
    importance = forest.get_feature_importance()
    for feat, score in importance.items():
        log(f"  {feat}: {score:.6f}")

    log("Step 12: Pheromone stats...")
    all_pher = [edata.get("pheromone", 1.0) for _, _, edata in G_fresh.edges(data=True)]
    high_pher = sum(1 for p in all_pher if p > 1.0)
    log(f"  Edges: {len(all_pher)}, Mean={np.mean(all_pher):.4f}, Std={np.std(all_pher):.4f}")
    log(f"  Min={np.min(all_pher):.4f}, Max={np.max(all_pher):.4f}")
    log(f"  Sparsity: {high_pher}/{len(all_pher)} ({100*high_pher/len(all_pher):.2f}%)")

    # Save JSON
    result = {
        "dataset": "tox21",
        "target": "NR-AhR",
        "n_samples": len(labels),
        "n_train": len(train_labels),
        "n_test": len(test_labels),
        "n_features": len(feature_df.columns),
        "n_trees": len(forest.trees_),
        "n_rules": len(forest.all_rules_),
        "fit_time_sec": fit_time,
        "train_accuracy": train_m["accuracy"],
        "train_f1": train_m["f1"],
        "test_accuracy_majority": test_m_maj["accuracy"],
        "test_balanced_accuracy_majority": test_m_maj["balanced_accuracy"],
        "test_precision_majority": test_m_maj["precision"],
        "test_recall_majority": test_m_maj["recall"],
        "test_f1_majority": test_m_maj["f1"],
        "test_auc_roc": test_m_maj.get("auc_roc", -1),
        "test_accuracy_weighted": test_m_wt["accuracy"],
        "test_balanced_accuracy_weighted": test_m_wt["balanced_accuracy"],
        "test_f1_weighted": test_m_wt["f1"],
        "pheromone_sparsity_pct": 100 * high_pher / len(all_pher),
        "top_rules": [
            {"rule": p.rule_string, "pheromone": p.avg_pheromone,
             "accuracy": p.accuracy, "coverage": p.coverage}
            for p in top_paths
        ],
        "feature_importance": dict(importance),
    }
    with open("output/tox21_nrahr_metrics.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    log("JSON saved to output/tox21_nrahr_metrics.json")
    log("ALL DONE")

except Exception as e:
    log(f"ERROR: {e}")
    log(traceback.format_exc())
