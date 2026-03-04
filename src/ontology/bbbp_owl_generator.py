"""
BBBP 데이터셋을 OWL 온톨로지로 변환
논문 SDT Phase 1: Ontology Construction
"""

from owlready2 import *
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
import pandas as pd
from pathlib import Path
from typing import Dict, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BBBPOntologyGenerator:
    """BBBP CSV → OWL/XML 변환기"""
    
    def __init__(self, output_path: str = "ontology/bbbp_ontology.owl"):
        self.output_path = output_path
        self.onto = None
        
    def create_schema(self):
        """온톨로지 스키마 생성 (Classes + Properties)"""
        
        logger.info("Creating ontology schema...")
        self.onto = get_ontology("http://example.org/bbbp.owl")
        
        with self.onto:
            # ===== CENTER CLASS =====
            class Molecule(Thing):
                """SDT의 중심 클래스"""
                pass
            
            # ===== FUNCTIONAL GROUP HIERARCHY =====
            class FunctionalGroup(Thing):
                pass
            
            class Amine(FunctionalGroup):
                pass
            
            class Alcohol(FunctionalGroup):
                pass
            
            class Carbonyl(FunctionalGroup):
                pass
            
            class Carboxyl(FunctionalGroup):
                pass
            
            class Ether(FunctionalGroup):
                pass
            
            class Halogen(FunctionalGroup):
                pass
            
            class Aromatic(FunctionalGroup):
                pass
            
            class Sulfur(FunctionalGroup):
                pass
            
            class Nitro(FunctionalGroup):
                pass
            
            class Amide(FunctionalGroup):
                pass
            
            # ===== OBJECT PROPERTIES =====
            class containsFunctionalGroup(ObjectProperty):
                """Molecule → FunctionalGroup"""
                domain = [Molecule]
                range = [FunctionalGroup]
            
            # ===== DATA PROPERTIES =====
            class hasSMILES(DataProperty, FunctionalProperty):
                domain = [Molecule]
                range = [str]
            
            class hasLabel(DataProperty, FunctionalProperty):
                """Target: 0 or 1"""
                domain = [Molecule]
                range = [int]
            
            class hasMolecularWeight(DataProperty, FunctionalProperty):
                domain = [Molecule]
                range = [float]
            
            class hasLogP(DataProperty, FunctionalProperty):
                domain = [Molecule]
                range = [float]
            
            class hasNumAtoms(DataProperty, FunctionalProperty):
                domain = [Molecule]
                range = [int]
            
            class hasNumHeavyAtoms(DataProperty, FunctionalProperty):
                domain = [Molecule]
                range = [int]
            
            class hasRingCount(DataProperty, FunctionalProperty):
                domain = [Molecule]
                range = [int]
            
            class hasAromaticRingCount(DataProperty, FunctionalProperty):
                domain = [Molecule]
                range = [int]
            
            class hasHBA(DataProperty, FunctionalProperty):
                domain = [Molecule]
                range = [int]
            
            class hasHBD(DataProperty, FunctionalProperty):
                domain = [Molecule]
                range = [int]
            
            class hasRotatableBonds(DataProperty, FunctionalProperty):
                domain = [Molecule]
                range = [int]
            
            class hasAromaticRing(DataProperty, FunctionalProperty):
                domain = [Molecule]
                range = [bool]
            
            class hasTPSA(DataProperty, FunctionalProperty):
                domain = [Molecule]
                range = [float]
            
            class obeysLipinskiRule(DataProperty, FunctionalProperty):
                domain = [Molecule]
                range = [bool]
        
        logger.info(f"✅ Schema created: {len(list(self.onto.classes()))} classes, "
                   f"{len(list(self.onto.data_properties()))} data properties")
    
    def detect_functional_groups(self, mol) -> Set[str]:
        """SMARTS 기반 기능기 탐지"""
        if mol is None:
            return set()
        
        groups = set()
        patterns = {
            'Amine': '[NX3;H2,H1;!$(NC=O)]',
            'Alcohol': '[OX2H]',
            'Carbonyl': '[CX3]=[OX1]',
            'Carboxyl': '[CX3](=O)[OX2H1]',
            'Ether': '[OD2]([#6])[#6]',
            'Halogen': '[F,Cl,Br,I]',
            'Aromatic': 'a',
            'Sulfur': '[#16]',
            'Nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
            'Amide': '[NX3][CX3](=[OX1])[#6]'
        }
        
        for name, smarts in patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                groups.add(name)
        
        return groups
    
    def extract_features(self, smiles: str) -> Dict:
        """RDKit로 분자 특성 추출"""
        # 염(salt) 제거
        if '.' in smiles:
            parts = smiles.split('.')
            smiles = max(parts, key=len)
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        features = {
            'mw': round(Descriptors.MolWt(mol), 2),
            'logp': round(Crippen.MolLogP(mol), 2),
            'atoms': mol.GetNumAtoms(),
            'heavy': mol.GetNumHeavyAtoms(),
            'rings': Descriptors.RingCount(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'rotatable': Descriptors.NumRotatableBonds(mol),
            'has_aromatic': Descriptors.NumAromaticRings(mol) > 0,
            'tpsa': round(Descriptors.TPSA(mol), 2),
            'functional_groups': self.detect_functional_groups(mol)
        }
        
        features['lipinski'] = (
            features['mw'] <= 500 and
            features['logp'] <= 5 and
            features['hba'] <= 10 and
            features['hbd'] <= 5
        )
        
        return features
    
    def create_instance(self, idx: int, smiles: str, label: int, features: Dict):
        """분자 인스턴스 생성"""
        with self.onto:
            mol = self.onto.Molecule(f"Mol_{idx:05d}")
            
            # Data properties (scalar values, not lists)
            mol.hasSMILES = smiles
            mol.hasLabel = label
            mol.hasMolecularWeight = features['mw']
            mol.hasLogP = features['logp']
            mol.hasNumAtoms = features['atoms']
            mol.hasNumHeavyAtoms = features['heavy']
            mol.hasRingCount = features['rings']
            mol.hasAromaticRingCount = features['aromatic_rings']
            mol.hasHBA = features['hba']
            mol.hasHBD = features['hbd']
            mol.hasRotatableBonds = features['rotatable']
            mol.hasAromaticRing = features['has_aromatic']
            mol.hasTPSA = features['tpsa']
            mol.obeysLipinskiRule = features['lipinski']
            
            # Object properties (functional groups)
            for fg_name in features['functional_groups']:
                fg_class = getattr(self.onto, fg_name, None)
                if fg_class:
                    mol.containsFunctionalGroup.append(fg_class)
    
    def generate_from_csv(self, csv_path: str, limit: int = None):
        """CSV → OWL 변환"""
        logger.info(f"Loading {csv_path}...")
        df = pd.read_csv(csv_path)
        
        if limit:
            df = df.head(limit)
        
        logger.info(f"Processing {len(df)} molecules...")
        
        self.create_schema()
        
        success = 0
        failed = 0
        
        for idx, row in df.iterrows():
            features = self.extract_features(row['smiles'])
            
            if features is None:
                failed += 1
                continue
            
            self.create_instance(idx, row['smiles'], int(row['p_np']), features)
            success += 1
            
            if (idx + 1) % 200 == 0:
                logger.info(f"  Progress: {idx + 1}/{len(df)}")
        
        logger.info(f"✅ Instances: {success} created, {failed} failed")
        
        # Save OWL file
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        self.onto.save(file=self.output_path, format="rdfxml")
        
        size_kb = Path(self.output_path).stat().st_size / 1024
        logger.info(f"💾 Saved: {self.output_path} ({size_kb:.1f} KB)")
        
        return self.onto


if __name__ == "__main__":
    generator = BBBPOntologyGenerator()
    onto = generator.generate_from_csv("data/bbbp/BBBP.csv")
    
    print("\n" + "="*60)
    print("OWL Ontology Generation Complete!")
    print(f"  Classes: {len(list(onto.classes()))}")
    print(f"  Instances: {len(list(onto.individuals()))}")
    print(f"  Object Properties: {len(list(onto.object_properties()))}")
    print(f"  Data Properties: {len(list(onto.data_properties()))}")
    print("="*60)
