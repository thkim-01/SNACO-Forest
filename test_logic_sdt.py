
from src.ontology.molecule_ontology import MoleculeOntology
from src.sdt.logic_learner import LogicSDTLearner

def test_logic_sdt():
    print("Initializing Ontology...")
    # Use a new file to avoid conflicts
    onto_manager = MoleculeOntology("test_logic_sdt.owl")
    
    # Create mock instances
    # Mol 1: Benzene ring + Alcohol -> Class 1
    features1 = {
        'molecular_weight': 100.0, 'num_atoms': 10, 'num_heavy_atoms': 6,
        'has_aromatic': True, 'num_aromatic_rings': 1,
        'functional_groups': ['Benzene', 'Alcohol'],
        'logp': 1.0, 'tpsa': 20.0, 
        'obeys_lipinski': True, 'mw_category': 'Low', 'logp_category': 'Moderate', 'tpsa_category': 'Low',
        'num_rings': 1, 'num_hba': 1, 'num_hbd': 1, 'num_rotatable_bonds': 0
    }
    mol1 = onto_manager.add_molecule_instance("Mol_Positive_1", features1, label=1)

    # Mol 2: Benzene ring only -> Class 0
    features2 = {
        'molecular_weight': 80.0, 'num_atoms': 6, 'num_heavy_atoms': 6,
        'has_aromatic': True, 'num_aromatic_rings': 1,
        'functional_groups': ['Benzene'],
        'logp': 2.0, 'tpsa': 0.0, 
        'obeys_lipinski': True, 'mw_category': 'Low', 'logp_category': 'Moderate', 'tpsa_category': 'Low',
        'num_rings': 1, 'num_hba': 0, 'num_hbd': 0, 'num_rotatable_bonds': 0
    }
    mol2 = onto_manager.add_molecule_instance("Mol_Negative_1", features2, label=0)
    
    # Mol 3: Alcohol only (no ring) -> Class 0
    features3 = {
        'molecular_weight': 46.0, 'num_atoms': 9, 'num_heavy_atoms': 3,
        'has_aromatic': False, 'num_aromatic_rings': 0,
        'functional_groups': ['Alcohol'],
        'logp': -0.5, 'tpsa': 20.0, 
        'obeys_lipinski': True, 'mw_category': 'Low', 'logp_category': 'Hydrophilic', 'tpsa_category': 'Low',
        'num_rings': 0, 'num_hba': 1, 'num_hbd': 1, 'num_rotatable_bonds': 0
    }
    mol3 = onto_manager.add_molecule_instance("Mol_Negative_2", features3, label=0)

    dataset = [mol1, mol2, mol3]
    print(f"Dataset size: {len(dataset)}")
    
    print("Training Logic SDT...")
    learner = LogicSDTLearner(onto_manager, max_depth=3, min_samples_split=2, min_samples_leaf=1, verbose=True)
    tree = learner.fit(dataset)
    
    print("Tree Traversal:")
    traverse(tree.root)
    
    # Verify separation
    # Expect split on 'hasFunctionalGroupRel.Alcohol' or similar
    # Mol1 has Alcohol (and Benzene) -> 1
    # Mol2 has Benzene (no Alcohol) -> 0
    # Mol3 has Alcohol (no Benzene) -> 0
    # Wait, Alcohol check: Mol1(Y), Mol3(Y). Benzene check: Mol1(Y), Mol2(Y).
    # Ideal rule: Has Benzene AND Has Alcohol -> 1. 
    # LogicSDT should find this combination (depth 2).

def traverse(node, indent=0):
    prefix = "  " * indent
    if node.is_leaf:
        print(f"{prefix}Leaf: Class {node.predicted_label} (n={node.num_instances})")
    else:
        print(f"{prefix}Node {node.node_id}: {node.refinement} (Gini={node.gini:.3f})")
        traverse(node.left_child, indent + 1)
        traverse(node.right_child, indent + 1)

if __name__ == "__main__":
    test_logic_sdt()
