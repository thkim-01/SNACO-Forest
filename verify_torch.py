import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.aco.semantic_forest import SemanticForest

import networkx as nx

def test_forest():
    print("Generating dummy data...")
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=42)
    feature_names = [f"f_{i}" for i in range(10)]
    df = pd.DataFrame(X, columns=feature_names)
    
    # Create simple taxonomy graph over features
    G = nx.DiGraph()
    G.add_node("ROOT")
    for fname in feature_names:
        G.add_edge("ROOT", fname)
    
    print("Initializing SemanticForest...")
    forest = SemanticForest(
        graph=G,
        n_trees=2,
        n_ants_per_tree=5,
        max_steps=5,
        criterion="entropy",
        torch_device="cpu"
    )
    
    print("Fitting model...")
    forest.fit(df, pd.Series(y))
    print("Fit completed successfully!")
    
    print("Predicting...")
    preds = forest.predict(df)
    print("Predictions shape:", preds.shape)
    print("Predictions unique:", np.unique(preds))
    print("Prediction test completed successfully!")

if __name__ == "__main__":
    test_forest()
