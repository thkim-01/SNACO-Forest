"""
Learner: SDT 학습 알고리즘
"""
import numpy as np
from typing import List, Optional
from .tree import SemanticDecisionTree, TreeNode
from .refinement import RefinementOperator, Refinement


class SDTLearner:
    """Semantic Decision Tree 학습기"""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 5,
                 min_samples_leaf: int = 2, class_weight: str = None, verbose: bool = True):
        """
        Args:
            max_depth: 트리의 최대 깊이
            min_samples_split: 분할을 수행하기 위한 최소 샘플 수
            min_samples_leaf: 리프 노드의 최소 샘플 수
            class_weight: 'balanced' or None. If 'balanced', apply class weights.
            verbose: 학습 과정 출력 여부
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.verbose = verbose
        
        self.tree = SemanticDecisionTree(max_depth, min_samples_split, min_samples_leaf)
        self.refinement_operator = RefinementOperator()
        self.class_weights_dict = None
    
    def fit(self, instances: List):
        """
        SDT 학습
        
        Args:
            instances: MolecularInstance 리스트
        """
        if self.verbose:
            print(f"Starting SDT training with {len(instances)} instances")
        
        # Calculate class weights if needed
        if self.class_weight == 'balanced':
            self.class_weights_dict = self._compute_class_weights(instances)
            if self.verbose:
                print(f"Class weights: {self.class_weights_dict}")
            
        # Root 노드 생성
        root_node = TreeNode(instances, depth=0, node_id=self.tree.get_next_node_id())
        self.tree.root = root_node
        self.tree.nodes.append(root_node)
        
        # 재귀적으로 트리 구축
        self._build_tree(root_node)
        
        if self.verbose:
            print(f"SDT training completed. Total nodes: {len(self.tree.nodes)}")
            
        return self.tree

    def _compute_class_weights(self, instances: List) -> dict:
        """Compute balanced class weights"""
        labels = [inst.label for inst in instances]
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
        return sum(self.class_weights_dict.get(inst.label, 1.0) for inst in instances)
        
        # 재귀적으로 트리 구축
        self._build_tree(root_node)
        
        if self.verbose:
            print(f"SDT training completed. Total nodes: {len(self.tree.nodes)}")
        
        return self.tree
    
    def _build_tree(self, node: TreeNode):
        """재귀적으로 트리 구축"""
        
        # Stopping criteria 체크
        if self._should_stop(node):
            self._make_leaf(node)
            return
        
        # 최적의 refinement 찾기
        best_refinement, best_gain = self._find_best_refinement(node)
        
        if best_refinement is None or best_gain <= 0:
            # 더 이상 분할할 수 없음
            self._make_leaf(node)
            return
        
        # Refinement 적용하여 분할
        node.set_refinement(best_refinement)
        left_instances, right_instances = self.refinement_operator.apply_refinement(
            node.instances, best_refinement
        )
        
        if self.verbose:
            print(f"Node {node.node_id} (depth {node.depth}): "
                  f"{best_refinement} -> "
                  f"Left: {len(left_instances)}, Right: {len(right_instances)}, "
                  f"Gain: {best_gain:.4f}")
        
        # 자식 노드 생성
        if len(left_instances) >= self.min_samples_leaf:
            node.left_child = TreeNode(
                left_instances,
                depth=node.depth + 1,
                node_id=self.tree.get_next_node_id()
            )
            self.tree.nodes.append(node.left_child)
            self._build_tree(node.left_child)
        
        if len(right_instances) >= self.min_samples_leaf:
            node.right_child = TreeNode(
                right_instances,
                depth=node.depth + 1,
                node_id=self.tree.get_next_node_id()
            )
            self.tree.nodes.append(node.right_child)
            self._build_tree(node.right_child)
        
        # 자식이 없으면 리프로 설정
        if node.left_child is None and node.right_child is None:
            self._make_leaf(node)
    
    def _should_stop(self, node: TreeNode) -> bool:
        """Stopping criteria 체크"""
        # 최대 깊이 도달
        if node.depth >= self.max_depth:
            return True
        
        # 샘플 수가 너무 적음
        if node.num_instances < self.min_samples_split:
            return True
        
        # 모든 인스턴스가 같은 레이블
        if len(node.label_counts) == 1:
            return True
        
        # Gini impurity가 0 (완벽하게 분리됨)
        if node.gini == 0:
            return True
        
        return False
    
    def _make_leaf(self, node: TreeNode):
        """리프 노드 생성"""
        # Majority voting
        predicted_label = max(node.label_counts, key=node.label_counts.get)
        node.set_as_leaf(predicted_label)
        
        if self.verbose:
            print(f"Leaf node {node.node_id}: "
                  f"Label {predicted_label}, "
                  f"Counts {node.label_counts}")
    
    def _find_best_refinement(self, node: TreeNode) -> tuple:
        """
        Gini gain이 가장 높은 refinement 찾기
        
        Returns:
            (best_refinement, best_gini_gain)
        """
        # 가능한 refinement 생성
        candidate_refinements = self.refinement_operator.generate_refinements(node.instances)
        
        if len(candidate_refinements) == 0:
            return None, 0.0
        
        best_refinement = None
        best_gain = -float('inf')
        
        # 각 refinement의 Gini gain 계산
        for refinement in candidate_refinements:
            gain = self._calculate_gini_gain(node, refinement)
            
            if gain > best_gain:
                best_gain = gain
                best_refinement = refinement
        
        return best_refinement, best_gain
    
    def _calculate_gini_gain(self, node: TreeNode, refinement: Refinement) -> float:
        """Gini gain 계산 (Weighted support)"""
        # 분할 수행
        left_instances, right_instances = self.refinement_operator.apply_refinement(
            node.instances, refinement
        )
        
        # 빈 분할은 유효하지 않음
        if len(left_instances) == 0 or len(right_instances) == 0:
            return 0.0
        
        # 최소 샘플 수 체크
        if len(left_instances) < self.min_samples_leaf or len(right_instances) < self.min_samples_leaf:
            return 0.0
        
        # 자식 노드의 Gini impurity 계산
        if self.class_weights_dict:
            n = self._get_total_weight(node.instances)
            n_left = self._get_total_weight(left_instances)
            n_right = self._get_total_weight(right_instances)
        else:
            n = node.num_instances
            n_left = len(left_instances)
            n_right = len(right_instances)
            
        if n == 0: return 0.0
        
        # 현재 노드 Gini impurity (Weighted if set)
        node_gini = self._calculate_gini(node.instances)
        # Note: We re-calculate node_gini here because node.gini stored in TreeNode might be unweighted
        # if TreeNode doesn't know about weights. Or we can just calculate it here to be safe and consistent.
        
        left_gini = self._calculate_gini(left_instances)
        right_gini = self._calculate_gini(right_instances)
        
        # Gini Gain = Parent Gini - Weighted Child Gini
        weighted_child_gini = (n_left / n) * left_gini + (n_right / n) * right_gini
        gini_gain = node_gini - weighted_child_gini
        
        return gini_gain
    
    def _calculate_gini(self, instances: List) -> float:
        """인스턴스 집합의 Gini impurity 계산 (Weighted support)"""
        if len(instances) == 0:
            return 0.0
        
        label_counts = {}
        for inst in instances:
            label_counts[inst.label] = label_counts.get(inst.label, 0) + 1
        
        # Apply weights
        if self.class_weights_dict:
            weighted_counts = {}
            for l, c in label_counts.items():
                weighted_counts[l] = c * self.class_weights_dict.get(l, 1.0)
            
            total = sum(weighted_counts.values())
            counts_to_use = weighted_counts
        else:
            total = len(instances)
            counts_to_use = label_counts
            
        if total == 0:
            return 0.0
        
        gini = 1.0
        for count in counts_to_use.values():
            if count > 0:
                p = count / total
                gini -= p * p
        
        return gini
