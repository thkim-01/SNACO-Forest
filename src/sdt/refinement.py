"""
Refinement: Semantic refinement 연산자 정의
"""
from typing import List, Tuple, Any


class Refinement:
    """
    Semantic Refinement를 나타내는 클래스
    형태: (property, operator, value)
    예: ('has_aromatic', '==', True)
    """
    
    def __init__(self, property_name: str, operator: str, value: Any):
        self.property = property_name
        self.operator = operator
        self.value = value
    
    def __repr__(self):
        return f"Refinement({self.property} {self.operator} {self.value})"
    
    def __eq__(self, other):
        return (self.property == other.property and 
                self.operator == other.operator and 
                self.value == other.value)
    
    def __hash__(self):
        return hash((self.property, self.operator, str(self.value)))
    
    def to_tuple(self) -> Tuple:
        """Tuple 형태로 변환"""
        return (self.property, self.operator, self.value)


class RefinementOperator:
    """
    Semantic Decision Tree를 위한 refinement 생성기
    """
    
    def __init__(self):
        # 정의 가능한 refinement 템플릿
        self.refinement_templates = []
        self._initialize_templates()
    
    def _initialize_templates(self):
        """
        가능한 모든 refinement 템플릿 정의
        SMILES에서 추출한 분자 특성 기반
        """
        
        # Boolean properties
        self.refinement_templates.extend([
            ('has_aromatic', '==', True),
            ('has_aromatic', '==', False),
            ('obeys_lipinski', '==', True),
            ('obeys_lipinski', '==', False),
        ])
        
        # Categorical properties
        mw_categories = ['Low', 'Medium', 'High']
        for cat in mw_categories:
            self.refinement_templates.append(('mw_category', '==', cat))
        
        logp_categories = ['Hydrophilic', 'Moderate', 'Lipophilic']
        for cat in logp_categories:
            self.refinement_templates.append(('logp_category', '==', cat))
        
        tpsa_categories = ['Low', 'Medium', 'High']
        for cat in tpsa_categories:
            self.refinement_templates.append(('tpsa_category', '==', cat))
        
        # Functional groups (contains)
        functional_groups = [
            'Amine', 'Alcohol', 'Carbonyl', 'Carboxyl', 'Ether',
            'Ester', 'Amide', 'Halogen', 'Aromatic', 'Sulfur', 'Nitro'
        ]
        for fg in functional_groups:
            self.refinement_templates.append(('functional_groups', 'contains', fg))
        
        # Numerical properties (threshold-based)
        # Ring count
        for i in range(0, 5):
            self.refinement_templates.extend([
                ('num_rings', '==', i),
                ('num_rings', '>', i),
            ])
        
        # Aromatic rings
        for i in range(0, 4):
            self.refinement_templates.extend([
                ('num_aromatic_rings', '==', i),
                ('num_aromatic_rings', '>', i),
            ])
        
        # H-bond acceptors
        for i in [0, 1, 2, 3, 5, 10]:
            self.refinement_templates.extend([
                ('num_hba', '>=', i),
                ('num_hba', '<=', i),
            ])
        
        # H-bond donors
        for i in [0, 1, 2, 3, 5]:
            self.refinement_templates.extend([
                ('num_hbd', '>=', i),
                ('num_hbd', '<=', i),
            ])
        
        # Heavy atoms
        for i in [5, 10, 15, 20, 30]:
            self.refinement_templates.extend([
                ('num_heavy_atoms', '>=', i),
                ('num_heavy_atoms', '<=', i),
            ])
        
        # Rotatable bonds
        for i in [0, 2, 5, 10]:
            self.refinement_templates.extend([
                ('num_rotatable_bonds', '>=', i),
                ('num_rotatable_bonds', '<=', i),
            ])
    
    def generate_refinements(self, instances: List) -> List[Refinement]:
        """
        주어진 인스턴스 집합에 대해 유효한 refinement 생성
        """
        valid_refinements = []
        
        for template in self.refinement_templates:
            refinement = Refinement(*template)
            
            # 최소한 일부 인스턴스가 만족하는 refinement만 유효
            satisfying_count = sum(
                1 for inst in instances 
                if inst.satisfies_refinement(refinement.to_tuple())
            )
            
            # 너무 극단적이지 않은 refinement만 선택
            # (최소 1개, 최대 전체-1개 만족)
            if 0 < satisfying_count < len(instances):
                valid_refinements.append(refinement)
        
        return valid_refinements
    
    def apply_refinement(self, instances: List, refinement: Refinement) -> Tuple[List, List]:
        """
        Refinement를 적용하여 인스턴스를 두 그룹으로 분할
        Returns: (satisfying_instances, non_satisfying_instances)
        """
        satisfying = []
        non_satisfying = []
        
        for inst in instances:
            if inst.satisfies_refinement(refinement.to_tuple()):
                satisfying.append(inst)
            else:
                non_satisfying.append(inst)
        
        return satisfying, non_satisfying
