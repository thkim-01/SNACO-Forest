import sys
from pathlib import Path
import json

# Add project root to sys.path so we can import src modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.aco.data_pipeline import ToxicityDataPipeline

def run_ablation():
    """
    Run parameter ablation tests for alpha (pig_alpha) and gamma (jump_gamma).
    Demonstrate that explicitly controlling these parameters directly changes
    the behavior of rule extraction on both Classification (BBBP) and Regression (FreeSolv).
    """
    print("="*70)
    print("   SNACO-Forest Parameter Ablation Study (Classification & Regression)")
    print("="*70)
    
    pipeline = ToxicityDataPipeline()
    
    # 1. Classification (BBBP) Ablation for Jump Gamma
    jump_gammas = [0.0, 1.0, 5.0]
    print("\n--- Testing jump_gamma (Ontology Penalty) on Classification (BBBP) ---")
    print(f"{'Gamma':<8} | {'Rules Generation Strategy':<30} | {'Test AUC':<10} | {'Test F1':<10}")
    for gamma in jump_gammas:
        res = pipeline.run("bbbp", jump_gamma=gamma, n_trees=5, n_generations=2, max_samples=500, skip_fit=False)
        if res.get("status") == "complete":
            auc = res.get("test_auc_roc", 0)
            f1 = res.get("test_f1_majority", 0)
            strategy = "Localized logic" if gamma > 0 else "Distal logic"
            print(f"{gamma:<8.1f} | {strategy:<30} | {auc:<10.4f} | {f1:<10.4f}")
        else:
            print(f"{gamma:<8.1f} | Error running pipeline")

    # 2. Regression (FreeSolv) Ablation for PIG Alpha
    pig_alphas = [0.1, 1.0, 3.0]
    print("\n--- Testing pig_alpha (Gain Regulation) on Regression (FreeSolv) ---")
    print(f"{'Alpha':<8} | {'Decision Bound Strictness':<30} | {'Test RMSE':<10} | {'Test R2':<10}")
    for alpha in pig_alphas:
        res = pipeline.run("freesolv", pig_alpha=alpha, n_trees=5, n_generations=2, max_samples=500, skip_fit=False)
        if res.get("status") == "complete":
            rmse = res.get("test_rmse", 0)
            r2 = res.get("test_r2", 0)
            strictness = "High Purity/Shallow" if alpha > 1.0 else "Deep Search"
            print(f"{alpha:<8.1f} | {strictness:<30} | {rmse:<10.4f} | {r2:<10.4f}")
        else:
            print(f"{alpha:<8.1f} | Error running pipeline")

if __name__ == "__main__":
    run_ablation()
