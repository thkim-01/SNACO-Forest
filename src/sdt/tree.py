"""
Tree: Semantic Decision Tree 구조 정의
"""
from typing import Optional, List
import numpy as np


class TreeNode:
    """SDT의 노드"""
    
    def __init__(self, instances: List, depth: int = 0, node_id: int = 0):
        self.instances = instances
        self.depth = depth
        self.node_id = node_id
        
        # 노드 정보
        self.refinement = None  # 이 노드의 refinement
        self.is_leaf = False
        self.predicted_label = None
        
        # 자식 노드
        self.left_child = None   # refinement를 만족하는 경우
        self.right_child = None  # refinement를 만족하지 않는 경우
        
        # 통계
        self.num_instances = len(instances)
        self.label_counts = self._count_labels()
        self.gini = self._calculate_gini()
        
    def _count_labels(self) -> dict:
        """레이블 분포 계산"""
        counts = {0: 0, 1: 0}
        for inst in self.instances:
            counts[inst.label] = counts.get(inst.label, 0) + 1
        return counts
    
    def _calculate_gini(self) -> float:
        """Gini impurity 계산 (CART 알고리즘)"""
        if self.num_instances == 0:
            return 0.0
        
        gini = 1.0
        for count in self.label_counts.values():
            if count > 0:
                p = count / self.num_instances
                gini -= p * p
        return gini
    
    def set_as_leaf(self, predicted_label: int):
        """리프 노드로 설정"""
        self.is_leaf = True
        self.predicted_label = predicted_label
    
    def set_refinement(self, refinement):
        """노드의 refinement 설정"""
        self.refinement = refinement
    
    def predict(self, instance) -> int:
        """단일 인스턴스에 대한 예측"""
        if self.is_leaf:
            return self.predicted_label
        
        # Refinement에 따라 자식 노드로 이동
        if instance.satisfies_refinement(self.refinement.to_tuple()):
            return self.left_child.predict(instance)
        else:
            return self.right_child.predict(instance)
    
    def get_info(self) -> dict:
        """노드 정보 반환"""
        return {
            'node_id': self.node_id,
            'depth': self.depth,
            'is_leaf': self.is_leaf,
            'num_instances': self.num_instances,
            'label_counts': self.label_counts,
            'gini': self.gini,
            'refinement': str(self.refinement) if self.refinement else None,
            'predicted_label': self.predicted_label
        }


class SemanticDecisionTree:
    """Semantic Decision Tree"""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 5, 
                 min_samples_leaf: int = 2):
        """
        Args:
            max_depth: 트리의 최대 깊이
            min_samples_split: 분할을 수행하기 위한 최소 샘플 수
            min_samples_leaf: 리프 노드의 최소 샘플 수
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        
        self.root = None
        self.node_counter = 0
        self.nodes = []  # 모든 노드 저장 (시각화용)
    
    def get_next_node_id(self) -> int:
        """다음 노드 ID 생성"""
        node_id = self.node_counter
        self.node_counter += 1
        return node_id
    
    def predict(self, instance) -> int:
        """단일 인스턴스 예측"""
        if self.root is None:
            raise ValueError("Tree has not been trained yet.")
        return self.root.predict(instance)
    
    def predict_batch(self, instances: List) -> np.ndarray:
        """여러 인스턴스에 대한 예측"""
        predictions = [self.predict(inst) for inst in instances]
        return np.array(predictions)
    
    def get_tree_structure(self) -> List[dict]:
        """트리 구조 정보 반환"""
        return [node.get_info() for node in self.nodes]
    
    def get_feature_importance(self) -> dict:
        """Feature importance 계산 (refinement별 Gini gain 기여도)"""
        importance = {}
        
        for node in self.nodes:
            if not node.is_leaf and node.refinement:
                prop = node.refinement.property
                
                # Gini gain 계산
                if node.left_child and node.right_child:
                    n = node.num_instances
                    n_left = node.left_child.num_instances
                    n_right = node.right_child.num_instances
                    
                    gini_gain = node.gini
                    gini_gain -= (n_left / n) * node.left_child.gini
                    gini_gain -= (n_right / n) * node.right_child.gini
                    
                    if prop not in importance:
                        importance[prop] = 0.0
                    importance[prop] += gini_gain
        
        # 정규화
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
