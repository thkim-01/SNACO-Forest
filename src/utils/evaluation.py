"""
Evaluation: 모델 평가 지표
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict


class ModelEvaluator:
    """모델 평가를 위한 클래스"""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_pred_proba: np.ndarray = None) -> Dict:
        """
        모델 평가 수행
        
        Args:
            y_true: 실제 레이블
            y_pred: 예측 레이블
            y_pred_proba: 예측 확률 (AUC-ROC 계산용, optional)
        
        Returns:
            평가 지표 딕셔너리
        """
        self.metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # AUC-ROC (확률 예측이 있는 경우)
        if y_pred_proba is not None:
            self.metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion Matrix
        self.metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return self.metrics
    
    def print_metrics(self):
        """평가 지표 출력"""
        print("\n" + "="*50)
        print("Model Evaluation Metrics")
        print("="*50)
        
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1-Score:  {self.metrics['f1_score']:.4f}")
        
        if 'auc_roc' in self.metrics:
            print(f"AUC-ROC:   {self.metrics['auc_roc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(self.metrics['confusion_matrix'])
        print("="*50 + "\n")
    
    def plot_confusion_matrix(self, save_path: str = None):
        """Confusion Matrix 시각화"""
        cm = self.metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                       save_path: str = None):
        """ROC Curve 시각화"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'SDT (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    def plot_feature_importance(self, importance_dict: Dict, 
                                top_n: int = 15, save_path: str = None):
        """Feature importance 시각화"""
        # 중요도 순으로 정렬
        sorted_importance = sorted(importance_dict.items(), 
                                   key=lambda x: x[1], reverse=True)[:top_n]
        
        features, importances = zip(*sorted_importance)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Information Gain (Normalized)')
        plt.title(f'Top {top_n} Feature Importance in SDT')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.close()


def calculate_probability_from_tree(tree, instance) -> float:
    """
    SDT에서 확률 예측 (리프 노드의 클래스 분포 기반)
    
    Args:
        tree: SemanticDecisionTree
        instance: MolecularInstance
    
    Returns:
        P(label=1) 확률
    """
    node = tree.root
    
    # 리프 노드까지 탐색
    while not node.is_leaf:
        if instance.satisfies_refinement(node.refinement.to_tuple()):
            node = node.left_child
        else:
            node = node.right_child
        
        # 자식이 없으면 현재 노드를 리프로 간주
        if node is None:
            break
    
    if node is None:
        return 0.5  # 기본값
    
    # 리프 노드의 클래스 분포로부터 확률 계산
    total = node.num_instances
    if total == 0:
        return 0.5
    
    count_positive = node.label_counts.get(1, 0)
    return count_positive / total
