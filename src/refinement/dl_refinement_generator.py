"""
논문 SDT의 핵심: Ontology 구조 기반 Refinement 자동 생성
Rule 1-5에 따라 DL Expression 생성
"""

from owlready2 import *
from typing import List, Dict, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class DLRefinement:
    """Description Logic Refinement 표현"""
    
    def __init__(self, refinement_type: str, property_name: str = None, 
                 target_class: str = None, value=None, operator: str = None):
        self.type = refinement_type
        self.property = property_name
        self.target = target_class
        self.value = value
        self.operator = operator
    
    def __repr__(self):
        if self.type == 'existential':
            return f"∃{self.property}.{self.target}"
        elif self.type == 'universal':
            return f"∀{self.property}.{self.target}"
        elif self.type == 'value':
            return f"{self.property}={self.value}"
        elif self.type == 'cardinality':
            return f"{self.property}{self.operator}{self.value}"
        return "Unknown"
    
    def __hash__(self):
        return hash((self.type, self.property, self.target, self.value, self.operator))
    
    def __eq__(self, other):
        return (self.type == other.type and 
                self.property == other.property and
                self.target == other.target and
                self.value == other.value and
                self.operator == other.operator)
    
    def to_owlready_restriction(self, onto):
        """Owlready2 제약조건으로 변환"""
        prop = getattr(onto, self.property, None)
        if prop is None:
            return None
        
        if self.type == 'existential':
            target_cls = getattr(onto, self.target, None)
            if target_cls:
                return prop.some(target_cls)
        
        elif self.type == 'value':
            return prop.value(self.value)
        
        elif self.type == 'cardinality':
            if self.operator == '>=':
                return prop.min(self.value)
            elif self.operator == '<=':
                return prop.max(self.value)
            elif self.operator == '==':
                return prop.exactly(self.value)
        
        return None


class RefinementGenerator:
    """온톨로지 구조 탐색 → Refinement 자동 생성"""
    
    def __init__(self, onto, dl_profile: str = "ALC"):
        self.onto = onto
        self.dl_profile = dl_profile
        self.allowed_types = self._allowed_types_for_profile(dl_profile)
    
    def generate_all_refinements(self, center_class_name: str = "Molecule") -> List[DLRefinement]:
        """
        논문 SDT의 Rule 1-5에 따라 모든 refinement 생성
        """
        refinements = []
        
        logger.info("Generating refinements...")
        
        # Rule 1: Existential Restrictions (∃ R.C)
        if 'existential' in self.allowed_types:
            existential = self._generate_existential_refinements(center_class_name)
            refinements.extend(existential)
            logger.info(f"  Existential: {len(existential)}")

        # Rule 3: Data Property Value / Cardinality Restrictions
        if 'cardinality' in self.allowed_types:
            data_prop = self._generate_data_property_refinements()
            refinements.extend(data_prop)
            logger.info(f"  Data Property: {len(data_prop)}")

        # Rule 4: Boolean / Value Restrictions
        if 'value' in self.allowed_types:
            boolean = self._generate_boolean_refinements()
            refinements.extend(boolean)
            logger.info(f"  Boolean: {len(boolean)}")
        
        logger.info(f"✅ Total refinements: {len(refinements)}")
        return refinements
    
    def _generate_existential_refinements(self, center_class: str) -> List[DLRefinement]:
        """Rule 1: ∃ hasFunctionalGroup.Amine"""
        refinements = []
        
        # containsFunctionalGroup 속성 찾기
        for obj_prop in self.onto.object_properties():
            if obj_prop.name == 'containsFunctionalGroup':
                # FunctionalGroup의 모든 서브클래스
                fg_class = getattr(self.onto, 'FunctionalGroup', None)
                if fg_class:
                    for subclass in fg_class.descendants():
                        if subclass != fg_class:  # 부모 제외
                            refinements.append(DLRefinement(
                                refinement_type='existential',
                                property_name=obj_prop.name,
                                target_class=subclass.name
                            ))
        
        return refinements
    
    def _generate_data_property_refinements(self) -> List[DLRefinement]:
        """Rule 3: hasRingCount >= 2"""
        refinements = []
        
        # Numeric properties
        numeric_props = {
            'hasRingCount': [0, 1, 2, 3],
            'hasAromaticRingCount': [0, 1, 2],
            'hasHBA': [1, 2, 3, 5],
            'hasHBD': [0, 1, 2, 3],
            'hasRotatableBonds': [2, 5, 10],
            'hasNumAtoms': [10, 20, 30],
            'hasNumHeavyAtoms': [10, 15, 20]
        }
        
        for prop_name, thresholds in numeric_props.items():
            for threshold in thresholds:
                refinements.append(DLRefinement(
                    refinement_type='cardinality',
                    property_name=prop_name,
                    value=threshold,
                    operator='>='
                ))
                refinements.append(DLRefinement(
                    refinement_type='cardinality',
                    property_name=prop_name,
                    value=threshold,
                    operator='<='
                ))
        
        return refinements
    
    def _generate_boolean_refinements(self) -> List[DLRefinement]:
        """Rule 4: hasAromaticRing = True"""
        refinements = []
        
        bool_props = ['hasAromaticRing', 'obeysLipinskiRule']
        
        for prop_name in bool_props:
            refinements.append(DLRefinement(
                refinement_type='value',
                property_name=prop_name,
                value=True,
                operator='=='
            ))
            refinements.append(DLRefinement(
                refinement_type='value',
                property_name=prop_name,
                value=False,
                operator='=='
            ))
        
        return refinements
    
    def filter_valid_refinements(self, refinements: List[DLRefinement], 
                                 instances: List) -> List[DLRefinement]:
        """
        인스턴스 집합에서 유효한 refinement만 필터링
        (최소 1개, 최대 전체-1개 만족)
        """
        valid = []
        
        for ref in refinements:
            # skip refinements not allowed by the DL profile
            if ref.type not in self.allowed_types:
                continue
            satisfying = self.count_satisfying_instances(ref, instances)
            if 0 < satisfying < len(instances):
                valid.append(ref)
        
        return valid

    def _allowed_types_for_profile(self, profile: str) -> List[str]:
        """Return allowed refinement types for a given DL profile.

        Supported profiles:
          - EL: existential, cardinality, value (no universal)
          - AL: existential, cardinality, value
          - ALC: existential, universal, cardinality, value
        """
        mapping = {
            'EL': ['existential', 'cardinality', 'value'],
            'AL': ['existential', 'cardinality', 'value'],
            'ALC': ['existential', 'universal', 'cardinality', 'value']
        }
        return mapping.get(profile.upper(), mapping['ALC'])
    
    def count_satisfying_instances(self, refinement: DLRefinement, 
                                   instances: List) -> int:
        """Refinement를 만족하는 인스턴스 개수"""
        count = 0
        
        for inst in instances:
            if self.instance_satisfies_refinement(inst, refinement):
                count += 1
        
        return count
    
    def instance_satisfies_refinement(self, instance, refinement: DLRefinement) -> bool:
        """인스턴스가 refinement를 만족하는지 확인"""
        
        if refinement.type == 'existential':
            # ∃ containsFunctionalGroup.Amine
            prop_values = getattr(instance, refinement.property, [])
            target_class = getattr(self.onto, refinement.target, None)
            
            if target_class:
                for val in prop_values:
                    if isinstance(val, target_class):
                        return True
            return False
        
        elif refinement.type == 'value':
            # hasAromaticRing = True
            prop_value = getattr(instance, refinement.property, None)
            return prop_value == refinement.value
        
        elif refinement.type == 'cardinality':
            # hasRingCount >= 2
            prop_value = getattr(instance, refinement.property, None)
            
            if prop_value is None:
                return False
            
            if refinement.operator == '>=':
                return prop_value >= refinement.value
            elif refinement.operator == '<=':
                return prop_value <= refinement.value
            elif refinement.operator == '==':
                return prop_value == refinement.value
        
        return False


if __name__ == "__main__":
    # 테스트
    from src.ontology.ontology_loader import OntologyLoader
    
    loader = OntologyLoader("ontology/bbbp_ontology.owl")
    onto = loader.load()
    
    generator = RefinementGenerator(onto)
    refinements = generator.generate_all_refinements()
    
    print(f"\nGenerated {len(refinements)} refinements")
    print("\nSample refinements:")
    for ref in refinements[:10]:
        print(f"  {ref}")
