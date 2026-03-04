# flake8: noqa

import argparse
import sys
import os
from pathlib import Path

# MUST BE FIRST: Initialize paths for all imports
from _init_paths import init_paths
if not init_paths():
    raise SystemExit(1)

import pandas as pd
import numpy as np
from src.ontology.molecule_ontology import MoleculeOntology
from src.ontology.smiles_converter import MolecularFeatureExtractor
from src.sdt.logic_forest import SemanticForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import logging
import json
from tabulate import tabulate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_forest(
    split_criterion: str = "information_gain",
    compute_backend: str = "auto",
    torch_device: str = "auto",
):
    logger.info(
        "Starting Semantic Forest Verification on BBBP... "
        f"(split_criterion={split_criterion})"
    )
    
    # Load BBBP
    df = pd.read_csv("data/bbbp/BBBP.csv")
    df_sample = df # Full dataset
    
    # Split
    train_df, test_df = train_test_split(df_sample, test_size=0.2, random_state=42)
    
    # Ontology
    onto_path = "ontology/temp_forest_bbbp.owl"
    if os.path.exists(onto_path): os.remove(onto_path)
    onto = MoleculeOntology(onto_path)
    extractor = MolecularFeatureExtractor()
    
    def populate(dataframe, name):
        instances = []
        for idx, row in dataframe.iterrows():
            try:
                feats = extractor.extract_features(row['smiles'])
                inst = onto.add_molecule_instance(f"Mol_{name}_{idx}", feats, label=int(row['p_np']))
                instances.append(inst)
            except: pass
        return instances
        
    logger.info("Populating Ontology...")
    train_instances = populate(train_df, "Train")
    test_instances = populate(test_df, "Test")
    
    # Load DL profile config (optional)
    config_path = Path(__file__).parent.joinpath('dl_profile_config.json')
    dl_profile = 'ALC'
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding='utf-8'))
            dl_profile = cfg.get('bbbp', dl_profile)
        except Exception:
            pass

    # Train Forest
    logger.info("Training Semantic Forest (5 estimators)...")
    # Using Depth 10 as it was best single tree
    forest = SemanticForest(
        onto,
        n_estimators=5,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=5,
        class_weight='balanced',
        verbose=True,
        learner_kwargs={
            'split_criterion': split_criterion,
            'compute_backend': compute_backend,
            'torch_device': torch_device,
        },
    )
    forest.fit(train_instances)
    
    # Evaluate
    logger.info("Evaluating...")
    probs = forest.predict_proba(test_instances)
    preds = forest.predict(test_instances)
    
    test_labels = [inst.hasLabel[0] for inst in test_instances]
    auc = roc_auc_score(test_labels, probs)
    acc = accuracy_score(test_labels, preds)
    
    # Print results as formatted table
    results_table = [
        {
            "Dataset": "BBBP",
            "Task": "p_np",
            "Search Strategy": search_strategy,
            "N Train": len(train_instances),
            "N Test": len(test_instances),
            "AUC": f"{auc:.4f}",
            "Accuracy": f"{acc:.4f}",
        }
    ]
    
    print("\n" + "="*120)
    print("SEMANTIC FOREST VERIFICATION RESULTS")
    print("="*120)
    print(tabulate(results_table, headers="keys", tablefmt="grid", showindex=False))
    print("="*120 + "\n")
    
    logger.info(f"Forest Results -> AUC: {auc:.4f}, Accuracy: {acc:.4f}")
    
    with open("forest_results.txt", "w") as f:
        f.write(f"Forest AUC: {auc:.4f}\n")
        f.write(f"Forest Accuracy: {acc:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Semantic Forest on BBBP")
    parser.add_argument(
        "--split-criterion",
        default="information_gain",
        choices=["information_gain", "id3", "gain_ratio", "c45_gain_ratio", "gini"],
        help="Split criterion: ID3(info gain), C4.5(gain ratio), CART(gini).",
    )
    parser.add_argument(
        "--algorithm",
        default=None,
        choices=["id3", "c45", "cart"],
        help=(
            "Algorithm alias for split criterion. "
            "id3 -> information_gain, c45 -> gain_ratio, cart -> gini. "
            "If provided, this overrides --split-criterion."
        ),
    )
    parser.add_argument(
        "--compute-backend",
        default="auto",
        choices=["auto", "numpy", "torch"],
        help="Compute backend for impurity calculations.",
    )
    parser.add_argument(
        "--torch-device",
        default="auto",
        help="Torch device when compute backend is torch/auto.",
    )
    args = parser.parse_args()

    if args.algorithm:
        args.split_criterion = {
            "id3": "information_gain",
            "c45": "gain_ratio",
            "cart": "gini",
        }[args.algorithm]

    verify_forest(
        split_criterion=args.split_criterion,
        compute_backend=args.compute_backend,
        torch_device=args.torch_device,
    )
