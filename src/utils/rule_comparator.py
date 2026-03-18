"""
Rule Comparator — 자동 생성 규칙과 고정/수동 규칙의 성능(AUC, F1, MAE 등)을 비교하는 모듈.
"""

import copy
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score

from src.aco.rule_extraction import DecisionPath, compute_class_weights, _majority_class
from src.aco.semantic_forest import SemanticForest, TreeResult

logger = logging.getLogger(__name__)


class RuleComparator:
    """자동 생성 규칙과 수동/고정 규칙의 앙상블 성능을 비교한다."""

    @staticmethod
    def _create_mock_forest_from_rules(
        rules: List[DecisionPath],
        feature_df: pd.DataFrame,
        labels: pd.Series,
        task: str = "classification"
    ) -> SemanticForest:
        """주어진 규칙만으로 동작하는 가상의 SemanticForest를 생성한다."""
        # 빈 그래프 생성
        import networkx as nx
        dummy_graph = nx.DiGraph()
        
        forest = SemanticForest(
            graph=dummy_graph,
            n_trees=1,
            task=task
        )
        
        n_samples = len(labels)
        label_array = labels.values.copy()
        
        if task == "regression":
            forest._default_prediction = float(np.mean(label_array.astype(float)))
            forest._class_weights = None
        else:
            forest._default_prediction = int(_majority_class(label_array))
            forest._class_weights = compute_class_weights(label_array)
            
        forest._feature_columns = list(feature_df.columns)
        
        # 가짜 트리 결과 생성
        dummy_tree = TreeResult(
            rules=copy.deepcopy(rules),
            oob_accuracy=1.0
        )
        forest.trees_ = [dummy_tree]
        forest.all_rules_ = copy.deepcopy(rules)
        forest.is_fitted_ = True
        return forest

    @classmethod
    def evaluate_rule_set(
        cls,
        rules: List[DecisionPath],
        feature_df: pd.DataFrame,
        labels: pd.Series,
        task: str = "classification"
    ) -> Dict[str, float]:
        """단일 규칙 집합의 앙상블 성능을 평가한다."""
        if not rules:
            return {}
            
        forest = cls._create_mock_forest_from_rules(rules, feature_df, labels, task)
        
        preds = forest.predict(feature_df, method="majority" if task == "classification" else "weighted")
        
        metrics = {}
        y_true = labels.values
        
        if task == "classification":
            metrics["accuracy"] = accuracy_score(y_true, preds)
            metrics["f1_score"] = f1_score(y_true, preds, average="weighted", zero_division=0)
            
            # ROC AUC (이진 분류만 지원)
            if len(np.unique(y_true)) == 2:
                proba = forest.predict_proba(feature_df)
                if proba.shape[1] > 1:
                    try:
                        metrics["auc"] = roc_auc_score(y_true, proba[:, 1])
                    except ValueError:
                        metrics["auc"] = float("nan")
        else:
            metrics["mae"] = mean_absolute_error(y_true, preds)
            
        return metrics

    @classmethod
    def generate_comparison_report(
        cls,
        auto_rules: List[DecisionPath],
        fixed_rules: List[DecisionPath],
        feature_df: pd.DataFrame,
        labels: pd.Series,
        task: str = "classification"
    ) -> str:
        """자동 생성 규칙과 고정 규칙의 성능 비교 리포트를 마크다운 형식으로 생성한다."""
        auto_metrics = cls.evaluate_rule_set(auto_rules, feature_df, labels, task)
        fixed_metrics = cls.evaluate_rule_set(fixed_rules, feature_df, labels, task)
        
        report = []
        report.append("# Rule Comparison Report\n")
        report.append(f"**Task Type**: {task.capitalize()}\n")
        report.append(f"- **Auto-generated Rules Count**: {len(auto_rules)}")
        report.append(f"- **Fixed/Manual Rules Count**: {len(fixed_rules)}\n")
        
        report.append("## Performance Metrics\n")
        report.append("| Metric | Auto-generated Rules | Fixed/Manual Rules | Difference (Fixed - Auto) |")
        report.append("|---|---|---|---|")
        
        all_metric_keys = set(auto_metrics.keys()).union(set(fixed_metrics.keys()))
        
        for metric in sorted(all_metric_keys):
            a_val = auto_metrics.get(metric, 0.0)
            f_val = fixed_metrics.get(metric, 0.0)
            diff = f_val - a_val
            
            # Formatting
            sign = "+" if diff > 0 else ""
            report.append(f"| **{metric.upper()}** | {a_val:.4f} | {f_val:.4f} | {sign}{diff:.4f} |")
            
        report.append("\n## Conclusion\n")
        if task == "classification":
            f1_diff = fixed_metrics.get("f1_score", 0.0) - auto_metrics.get("f1_score", 0.0)
            if f1_diff > 0:
                report.append("Fixed rules **improved** the overall F1 score.")
            elif f1_diff < 0:
                report.append("Fixed rules **decreased** the overall F1 score.")
            else:
                report.append("Fixed rules showed **no difference** in F1 score.")
        else:
            mae_diff = fixed_metrics.get("mae", 0.0) - auto_metrics.get("mae", 0.0)
            if mae_diff < 0:
                report.append("Fixed rules **improved** the MAE (lower error).")
            elif mae_diff > 0:
                report.append("Fixed rules **worsened** the MAE.")
            else:
                report.append("Fixed rules showed **no difference** in MAE.")
                
        return "\n".join(report)
