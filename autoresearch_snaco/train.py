"""
Autoresearch SNACO-Forest tuning script.
Usage: python train.py

The AI researcher agent modifies the Hyperparameters below to maximize PRC-AUC
across a 5-minute fixed time budget.
"""

import sys
import time
from pathlib import Path

# Setup local project path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.aco.data_pipeline import ToxicityDataPipeline

# ---------------------------------------------------------------------------
# Hyperparameters (Agent edits these to optimize performance)
# ---------------------------------------------------------------------------

# Primary Dataset for Optimization (Default: BBBP)
TARGET_DATASET = "bbbp"

# Decision Logic Bounds
PIG_ALPHA = 1.0           # Controls Information Gain greediness
JUMP_GAMMA = 0.5          # Controls Semantic Tree spatial constraints (locality)
MIN_PHEROMONE = 0.05      # Global minimum pheromone floor
EVAPORATION_RATE = 0.1    # Ratio of pheromones dying per generation

# Forest Volume Setup
N_TREES = 3               # Number of parallel trees (Bagging)
N_GENERATIONS = 2         # Pheromone convergence loops per tree
N_ANTS_PER_TREE = 10      # Meta-heuristic scaling particles

# Data sampling constraint (controls execution time!)
# Important: Higher max_samples gives better metrics but takes longer. 
# Keep within 5 minute budget!
MAX_SAMPLES = 800

# ---------------------------------------------------------------------------
# Execution and Evaluation Loop
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    
    print(f"Initializing SNACO-Forest Autoresearch Run on '{TARGET_DATASET}'")
    print("="*60)
    
    pipeline = ToxicityDataPipeline()
    try:
        # We pass the modified hyperparameters to the pipeline executor
        result = pipeline.run(
            dataset_name=TARGET_DATASET,
            n_trees=N_TREES,
            n_generations=N_GENERATIONS,
            n_ants_per_tree=N_ANTS_PER_TREE,
            pig_alpha=PIG_ALPHA,
            jump_gamma=JUMP_GAMMA,
            evaporation_rate=EVAPORATION_RATE,
            min_pheromone=MIN_PHEROMONE,
            max_samples=MAX_SAMPLES,
            skip_fit=False
        )
        
        t_end = time.time()
        
        # Determine performance
        if TARGET_DATASET in ["esol", "freesolv", "lipophilicity"]:
            # For Standard Regression: Use RMSE
            val_metric = result.get("test_rmse", 999.0)
            metric_name = "test_rmse"
        elif TARGET_DATASET in ["qm7", "qm8", "qm9"]:
            # For Quantum Mechanics Regression: Use MAE
            val_metric = result.get("test_mae", 999.0)
            metric_name = "test_mae"
        elif TARGET_DATASET.lower() == "muv":
            # For Extreme Imbalance (MUV): Use PRC-AUC
            val_metric = result.get("test_prc_auc", 0.0)
            metric_name = "test_prc_auc"
        else:
            # For General Classification: Use AUC-ROC
            val_metric = result.get("test_auc_roc", 0.0)
            metric_name = "test_auc_roc"
            
        print("="*60)
        print("--- RUN COMPLETE ---")
        print(f"total_seconds:    {t_end - t_start:.1f}")
        print(f"{metric_name}:      {val_metric:.6f}")
        print("---")
        if "trained_forest" in result and result["trained_forest"].all_rules_:
            for i, rule in enumerate(result["trained_forest"].all_rules_[:2]):
                rule_str = rule.to_rule_string() if hasattr(rule, "to_rule_string") else str(rule)
                print(f"Top Rule {i+1}: {rule_str}")
        print("---")
        
        # Karpathy autoresearch often parses `val_bpb: {float}`, so for backwards 
        # compatibility with standard parsing loops we also output it as a proxy.
        # But we invert accuracy metrics so the agent learning to "minimize val_bpb" actually maximizes accuracy.
        if metric_name in ["test_auc_roc", "test_prc_auc"]:
            pseudo_bpb = 1.0 - val_metric
        else:
            pseudo_bpb = val_metric
            
        print(f"val_bpb:          {pseudo_bpb:.6f}")
        
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
