
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_scaffold(smiles, include_chirality=False):
    """
    Compute the Bemis-Murcko scaffold for a SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def scaffold_split(smiles_list, labels, test_size=0.2, seed=42, log_every_n=1000):
    """
    Split a dataset based on Bemis-Murcko scaffolds.
    
    Args:
        smiles_list (list): List of SMILES strings.
        labels (list): List of labels corresponding to the SMILES.
        test_size (float): Fraction of the dataset to be used as test set.
        seed (int): Random seed for reproducibility.
        log_every_n (int): Log progress every n molecules.
        
    Returns:
        tuple: (train_smiles, test_smiles, train_labels, test_labels)
    """
    logger.info(f"Generating scaffolds for {len(smiles_list)} molecules...")
    
    scaffolds = defaultdict(list)
    for i, (smiles, label) in enumerate(zip(smiles_list, labels)):
        scaffold = generate_scaffold(smiles)
        if scaffold is None:
            logger.warning(f"Could not generate scaffold for {smiles}. Using empty scaffold.")
            scaffold = ""
        scaffolds[scaffold].append((i, smiles, label))
        
        if (i + 1) % log_every_n == 0:
            logger.info(f"Processed {i + 1} molecules")
            
    # Sort scaffolds by size (largest to smallest)
    scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)
    
    # Shuffle scaffold sets to ensure random distribution of scaffold sizes/types in train/test
    # while keeping molecules of same scaffold together.
    # This avoids the issue where test set gets only singletons or specific classes.
    rng = np.random.RandomState(seed)
    # Convert to list of lists to shuffle
    scaffold_sets = list(scaffold_sets)
    rng.shuffle(scaffold_sets)
    
    train_size = int(len(smiles_list) * (1 - test_size))
    
    train_indices = []
    test_indices = []
    
    train_counts = 0
    
    # Assign scaffolds to train or test
    # Note: A common strategy is to put the expected test_size fraction into test, trying to balance or just filling train first.
    # Standard scaffold split (like in DeepChem/Chemprop) usually fills train set with largest scaffolds first
    # to ensure the model sees diverse scaffolds, but the test set will have scaffolds NOT seen in train (mostly).
    # This tests generalization to new scaffolds.
    
    for scaffold_set in scaffold_sets:
        if train_counts + len(scaffold_set) <= train_size:
            train_indices.extend([x[0] for x in scaffold_set])
            train_counts += len(scaffold_set)
        else:
            test_indices.extend([x[0] for x in scaffold_set])
            
    # Map back to lists
    train_smiles = [smiles_list[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_smiles = [smiles_list[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]
    
    # Shuffle train set (optional but good practice)
    # We do NOT shuffle test set to keep order if needed, but it doesn't matter much.
    # Actually, let's shuffle both to avoid any artifact of scaffold sorting order in the final lists
    rng = np.random.RandomState(seed)
    
    train_perm = rng.permutation(len(train_smiles))
    train_smiles = [train_smiles[i] for i in train_perm]
    train_labels = [train_labels[i] for i in train_perm]
    
    test_perm = rng.permutation(len(test_smiles))
    test_smiles = [test_smiles[i] for i in test_perm]
    test_labels = [test_labels[i] for i in test_perm]
    
    logger.info(f"Scaffold split complete.")
    logger.info(f"Train samples: {len(train_smiles)}, Test samples: {len(test_smiles)}")
    
    return train_smiles, test_smiles, train_labels, test_labels
