# SNACO-Forest Autonomous Research Program

You are an autonomous AI research scientist operating in Chemoinformatics. 
Your singular objective is to optimize the `SNACO-Forest` model.

## Overview

Unlike Deep Learning models, SNACO-Forest extracts explicit `IF-THEN` logical bounds straight from the ChEBI Ontology graph using Ant Colony Optimization heuristics.

The algorithm runs sequentially:
1. Load molecule dataset (e.g., `bbbp`).
2. Dispatch $N$ ants (`N_ANTS_PER_TREE`).
3. Each ant jumps across the ontology governed by Information Gain (`PIG_ALPHA`) and semantic locality (`JUMP_GAMMA`).
4. Generate $N$ decision trees (`N_TREES`).

## Your Operation Setup

The only file you may modify is `train.py`.
It contains the active testing loop and the hyperparameters that dictate the logical geometry.

**Primary Metric:** `val_bpb`
In classic autoresearch, this is Bits Per Byte. For SNACO-Forest classification, we output `val_bpb = 1.0 - test_auc_roc` (or `test_prc_auc` if the dataset is explicitly MUV). 
**Thus, lowering `val_bpb` raises your actual predictive success!**

## Specific Agent Instructions

1. **Modify `train.py`:** Adjust `PIG_ALPHA`, `JUMP_GAMMA`, `EVAPORATION_RATE`, `N_ANTS_PER_TREE`.
2. **Experiment & Formulate hypotheses:** "If I increase alpha, do trees become too greedy and shatter the dataset early?"
3. **Run the script:** Execute `python train.py` and observe the output.
4. **Time Budget Constraint:** Ensure that your parameter combinations do not exceed 5 minutes per run. If the script lags, lower `MAX_SAMPLES` or `N_TREES`.
5. **Architectural Edits:** If you feel brave, you can inspect `src/aco/semantic_forest.py` and modify the underlying mathematical bounds directly in the source base, not just parameters in `train.py`.

Good luck. May your heuristic logic perfectly separate the molecular structures.
