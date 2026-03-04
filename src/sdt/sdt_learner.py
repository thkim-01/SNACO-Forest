"""
논문 SDT 학습 알고리즘
Reasoner 기반 Center Class 동적 갱신 + DL Refinement 적용
"""

from owlready2 import *
from typing import List, Dict, Tuple
import numpy as np
import types
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.refinement.dl_refinement_generator import RefinementGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SDTNode:
    """논문 SDT의 노드 (Center Class 기반)"""
    
    def __init__(self, center_class, instances: List, depth: int = 0, node_id: int = 0):
        self.center_class = center_class
        self.instances = instances
        self.depth = depth
        self.node_id = node_id
        
        self.refinement = None
        self.is_leaf = False
        self.predicted_label = None
        
        self.left_child = None  # refinement 만족
        self.right_child = None  # refinement 불만족
        
        self.num_instances = len(instances)
        self.label_counts = self._count_labels()
        self.gini = self._calculate_gini()
    
    def _count_labels(self) -> Dict[int, int]:
        """레이블 분포"""
        counts = {0: 0, 1: 0}
        for inst in self.instances:
            label = getattr(inst, 'hasLabel', None)
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
        return counts
    
    def _calculate_gini(self) -> float:
        """Gini impurity 계산"""
        if self.num_instances == 0:
            return 0.0
        
        gini = 1.0
        for count in self.label_counts.values():
            if count > 0:
                p = count / self.num_instances
                gini -= p * p
        return gini


class SemanticDecisionTreeLearner:
    """논문 SDT 학습 알고리즘"""
    
    def __init__(self, onto, max_depth: int = 10, 
                 min_samples_split: int = 10, min_samples_leaf: int = 5,
                 class_weight: str = None,
                 verbose: bool = True,
                 dl_profile: str = "ALC"):
        self.onto = onto
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.verbose = verbose
        
        self.refinement_generator = RefinementGenerator(onto, dl_profile)
        self.root = None
        self.node_counter = 0
        self.nodes = []
        self.class_weights_dict = None
    
    def fit(self, center_class_name: str = "Molecule"):
        """SDT 학습"""
        logger.info(f"Starting SDT training with center class: {center_class_name}")
        
        # Center class와 인스턴스 가져오기
        center_class = getattr(self.onto, center_class_name)
        instances = list(center_class.instances())
        
        logger.info(f"Total instances: {len(instances)}")

        # Calculate class weights if needed
        if self.class_weight == 'balanced':
            self.class_weights_dict = self._compute_class_weights(instances)
            logger.info(f"Class weights: {self.class_weights_dict}")
        
        # Root 노드 생성
        self.root = SDTNode(center_class, instances, depth=0, node_id=self._get_node_id())
        self.nodes.append(self.root)
        
        # 재귀적 트리 구축
        self._build_tree(self.root)
        
        logger.info(f"✅ SDT training complete. Total nodes: {len(self.nodes)}")
        
        return self.root

    def _compute_class_weights(self, instances: List) -> Dict[int, float]:
        """Compute balanced class weights"""
        labels = []
        for inst in instances:
            l = getattr(inst, 'hasLabel', None)
            if l is not None:
                labels.append(l)
        
        if not labels:
            return {}
            
        unique_classes, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        n_classes = len(unique_classes)
        
        weights = {}
        for cls, count in zip(unique_classes, counts):
            weights[cls] = total / (n_classes * count)
            
        return weights

    def _get_total_weight(self, instances: List) -> float:
        """Calculate total weight of instances"""
        if not self.class_weights_dict:
            return float(len(instances))
            
        total_weight = 0.0
        for inst in instances:
            l = getattr(inst, 'hasLabel', None)
            if l is not None:
                total_weight += self.class_weights_dict.get(l, 1.0)
        return total_weight
    
    def _get_node_id(self) -> int:
        """노드 ID 생성"""
        node_id = self.node_counter
        self.node_counter += 1
        return node_id
    
    def _build_tree(self, node: SDTNode):
        """재귀적 트리 구축"""
        
        # Stopping criteria
        if self._should_stop(node):
            self._make_leaf(node)
            return
        
        # Refinement 생성 및 최적 선택
        best_refinement, best_gain, left_instances, right_instances = self._find_best_refinement(node)
        
        if best_refinement is None or best_gain <= 0:
            self._make_leaf(node)
            return
        
        # Refinement 적용
        node.refinement = best_refinement
        
        if self.verbose:
            logger.info(f"Node {node.node_id} (depth {node.depth}): {best_refinement} "
                       f"-> Left: {len(left_instances)}, Right: {len(right_instances)}, "
                       f"Gain: {best_gain:.4f}")
        
        # 자식 노드 생성
        if len(left_instances) >= self.min_samples_leaf:
            node.left_child = SDTNode(
                node.center_class, left_instances,
                depth=node.depth + 1, node_id=self._get_node_id()
            )
            self.nodes.append(node.left_child)
            self._build_tree(node.left_child)
        
        if len(right_instances) >= self.min_samples_leaf:
            node.right_child = SDTNode(
                node.center_class, right_instances,
                depth=node.depth + 1, node_id=self._get_node_id()
            )
            self.nodes.append(node.right_child)
            self._build_tree(node.right_child)
        
        # 자식이 없으면 leaf
        if node.left_child is None and node.right_child is None:
            self._make_leaf(node)
    
    def _should_stop(self, node: SDTNode) -> bool:
        """Stopping criteria"""
        if node.depth >= self.max_depth:
            return True
        if node.num_instances < self.min_samples_split:
            return True
        if len(node.label_counts) == 1:
            return True
        if node.gini == 0:
            return True
        return False
    
    def _make_leaf(self, node: SDTNode):
        """Leaf 노드 생성"""
        node.is_leaf = True
        node.predicted_label = max(node.label_counts, key=node.label_counts.get)
        
        if self.verbose:
            logger.info(f"Leaf {node.node_id}: Label {node.predicted_label}, "
                       f"Counts {node.label_counts}")
    
    def _find_best_refinement(self, node: SDTNode) -> Tuple:
        """최적 refinement 찾기"""
        
        # 모든 refinement 생성
        all_refinements = self.refinement_generator.generate_all_refinements()
        
        # 유효한 refinement 필터링
        valid_refinements = self.refinement_generator.filter_valid_refinements(
            all_refinements, node.instances
        )
        
        if len(valid_refinements) == 0:
            return None, 0.0, [], []
        
        best_refinement = None
        best_gain = -float('inf')
        best_left = []
        best_right = []
        
        # 각 refinement의 정보 이득 계산
        for refinement in valid_refinements:
            left_instances = []
            right_instances = []
            
            for inst in node.instances:
                if self.refinement_generator.instance_satisfies_refinement(inst, refinement):
                    left_instances.append(inst)
                else:
                    right_instances.append(inst)
            
            # 최소 샘플 수 체크
            if len(left_instances) < self.min_samples_leaf or len(right_instances) < self.min_samples_leaf:
                continue
            
            # Gini Gain 계산
            gain = self._calculate_gini_gain(
                node.instances, left_instances, right_instances
            )
            
            if gain > best_gain:
                best_gain = gain
                best_refinement = refinement
                best_left = left_instances
                best_right = right_instances
        
        return best_refinement, best_gain, best_left, best_right
    
    def _calculate_gini_gain(self, parent_instances: List, 
                                    left_instances: List, right_instances: List) -> float:
        """Gini gain 계산 (Weighted supported)"""
        parent_gini = self._calculate_gini(parent_instances)
        
        if self.class_weights_dict:
            n = self._get_total_weight(parent_instances)
            n_left = self._get_total_weight(left_instances)
            n_right = self._get_total_weight(right_instances)
        else:
            n = len(parent_instances)
            n_left = len(left_instances)
            n_right = len(right_instances)
            
        if n == 0: return 0.0
        
        left_gini = self._calculate_gini(left_instances)
        right_gini = self._calculate_gini(right_instances)
        
        weighted_gini = (n_left / n) * left_gini + (n_right / n) * right_gini
        
        return parent_gini - weighted_gini
    
    def _calculate_gini(self, instances: List) -> float:
        """Gini impurity 계산 (Weighted supported)"""
        if len(instances) == 0:
            return 0.0
        
        label_counts = {}
        for inst in instances:
            label = getattr(inst, 'hasLabel', None)
            if label is not None:
                label_counts[label] = label_counts.get(label, 0) + 1
        
        # Apply weights
        if self.class_weights_dict:
            weighted_counts = {}
            for l, c in label_counts.items():
                weighted_counts[l] = c * self.class_weights_dict.get(l, 1.0)
            
            total = sum(weighted_counts.values())
            counts_to_use = weighted_counts
        else:
            total = len(instances) # Or sum(label_counts.values())
            counts_to_use = label_counts
            
        if total == 0:
            return 0.0
        
        gini = 1.0
        for count in counts_to_use.values():
            if count > 0:
                p = count / total
                gini -= p * p
        
        return gini
    
    def predict(self, instance) -> int:
        """단일 인스턴스 예측"""
        node = self.root
        
        while not node.is_leaf:
            if self.refinement_generator.instance_satisfies_refinement(instance, node.refinement):
                node = node.left_child
            else:
                node = node.right_child
            
            if node is None:
                break
        
        if node and node.is_leaf:
            return node.predicted_label
        
        return 0  # default
    
    def predict_batch(self, instances: List) -> np.ndarray:
        """여러 인스턴스 예측"""
        return np.array([self.predict(inst) for inst in instances])


if __name__ == "__main__":
    from src.ontology.ontology_loader import OntologyLoader
    
    # OWL 로드
    loader = OntologyLoader("ontology/bbbp_ontology.owl")
    onto = loader.load()
    
    # SDT 학습
    learner = SemanticDecisionTreeLearner(onto, max_depth=5, verbose=True)
    root = learner.fit("Molecule")
    
    print(f"\n✅ Training complete!")
    print(f"   Total nodes: {len(learner.nodes)}")
    print(f"   Max depth: {max(n.depth for n in learner.nodes)}")
