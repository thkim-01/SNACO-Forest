import pandas as pd
from pathlib import Path
import json

def generate_table_stats():
    config_path = Path("configs/dataset_ontology_config.json")
    with open(config_path) as f:
        config = json.load(f)
        
    print("="*60)
    print(f"{'Dataset':<15} | {'Samples':<10} | {'Type':<15} | {'Pos Ratio (%)':<15}")
    print("="*60)
    
    for name, cfg in config.items():
        if "file" not in cfg:
            continue
            
        csv_path = Path(cfg["file"])
        if not csv_path.exists():
            continue
            
        try:
            df = pd.read_csv(csv_path)
            target = cfg["label_col"]
            
            if target in df.columns:
                valid_df = df.dropna(subset=[target])
                samples = len(valid_df)
                task_type = cfg.get("task", "classification")
                
                if task_type == "classification":
                    pos_ratio = (valid_df[target] == 1).sum() / samples * 100
                    ratio_str = f"{pos_ratio:.1f}%"
                else:
                    ratio_str = "N/A (Reg)"
                    
                type_str = task_type.capitalize()
                print(f"{name:<15} | {samples:<10} | {type_str:<15} | {ratio_str:<15}")
        except Exception as e:
            print(f"{name:<15} | Error: {e}")
            
    print("="*60)

if __name__ == "__main__":
    print("Extracting Dataset Imbalance Stats for Figure/Table generation...\n")
    generate_table_stats()
