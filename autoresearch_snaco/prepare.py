"""
autoresearch_snaco prepare script.
Downloads and caches bioassay datasets and ChEBI/DTO ontology structures.
This file is executed once before the autonomous loop starts. Do not modify.
"""

import sys
from pathlib import Path
import json
import time

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_processor import DatasetProcessor
from src.ontology.ontology_graph import OntologyGraph
import rdkit

def prepare_data():
    print("Pre-warming dataset configurations and ontology graphs...")
    t0 = time.time()
    
    config_path = project_root / "configs" / "dataset_ontology_config.json"
    with open(config_path) as f:
        config = json.load(f)
        
    print(f"Loaded {len(config)} dataset configurations.")
    
    # Try touching one graph to ensure caching
    try:
        onto = OntologyGraph()
        print(f"Ontology initialized. Total Nodes: {len(onto.graph.nodes)}")
    except Exception as e:
        print(f"Ontology initialization skipped or failed: {e}")
        
    print(f"Preparation complete in {time.time()-t0:.2f}s.")

if __name__ == "__main__":
    prepare_data()
