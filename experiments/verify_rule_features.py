"""
간단한 데이터셋으로 Rule 추출/시각화 강화 기능들을 검증하는 스크립트.
"""
import sys
import os
import networkx as nx
import pandas as pd
import numpy as np

# ROOT directory 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.aco.semantic_forest import SemanticForest
from src.utils.rule_io import RuleExporter
from src.utils.rule_comparator import RuleComparator

def main():
    print("=== Rule Features Verification ===")
    
    # 1. Dummy Graph and Data
    G = nx.DiGraph()
    G.add_node("FeatureA", is_feature=True, feature_key="FeatureA")
    G.add_node("FeatureB", is_feature=True, feature_key="FeatureB")
    G.add_edge("Root", "FeatureA")
    G.add_edge("Root", "FeatureB")
    
    np.random.seed(42)
    df = pd.DataFrame({
        "FeatureA": np.random.rand(100),
        "FeatureB": np.random.rand(100),
    })
    # Labels: 1 if FeatureA > 0.5 and FeatureB > 0.5, else 0
    labels = pd.Series(((df["FeatureA"] > 0.5) & (df["FeatureB"] > 0.5)).astype(int))
    
    print("\n--- Training Auto Forest ---")
    forest_auto = SemanticForest(
        graph=G, n_trees=3, n_ants_per_tree=10, 
        n_generations=2, seed=42
    )
    forest_auto.fit(df, labels)
    
    # 2. Rule Export
    print("\n--- Testing Rule Export ---")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "fixed_rules")
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "test_rules.json")
    csv_path = os.path.join(output_dir, "test_rules.csv")
    md_path = os.path.join(output_dir, "test_rules.md")
    txt_path = os.path.join(output_dir, "test_rules.txt")
    
    forest_auto.export_rules(json_path, format="json", top_k=5)
    forest_auto.export_rules(csv_path, format="csv", top_k=5)
    forest_auto.export_rules(md_path, format="md", top_k=5)
    forest_auto.export_rules(txt_path, format="txt", top_k=5)
    
    print(f"Rules exported to {output_dir}")
    
    # 3. Rule Fixing (Loading & Injection)
    print("\n--- Testing Rule Fixing ---")
    fixed_rules = RuleExporter.load_fixed_rules(json_path)
    
    forest_fixed = SemanticForest(
        graph=G, n_trees=3, n_ants_per_tree=10,
        n_generations=2, seed=123, fixed_rules=fixed_rules
    )
    forest_fixed.fit(df, labels)
    print(f"Fixed rules length in forest: {len(forest_fixed.fixed_rules)}")
    print(f"Total rules extracted including dummy tree: {len(forest_fixed.all_rules_)}")
    
    # 4. Rule Comparison
    print("\n--- Testing Rule Comparison ---")
    report = RuleComparator.generate_comparison_report(
        auto_rules=forest_auto.all_rules_,
        fixed_rules=forest_fixed.all_rules_,
        feature_df=df,
        labels=labels,
        task="classification"
    )
    print("\n" + report)

if __name__ == "__main__":
    main()
