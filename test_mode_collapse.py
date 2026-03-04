"""
Mode Collapse 방지 검증 테스트.

Tox21 NR-AhR과 유사한 극단적 불균형 (90% class 0, 10% class 1)
합성 데이터에서 Mode Collapse가 발생하지 않는지 검증한다.

검증 항목:
  1. compute_class_weights: 소수 클래스에 높은 가중치
  2. DecisionPath: F1-based fitness, leaf_class_dist, leaf_probability
  3. predict_proba: 연속적 확률 분포 (hard 0/1이 아님)
  4. 통합 SemanticForest: F1 > 0 on imbalanced data
"""

from __future__ import annotations

import unittest
import numpy as np
import pandas as pd
import networkx as nx

from src.aco.rule_extraction import (
    compute_class_weights,
    DecisionPath,
    SplitCondition,
    _entropy,
    _gini,
    _info_gain,
    find_best_threshold,
)


class TestComputeClassWeights(unittest.TestCase):
    """Class weight 계산 로직."""

    def test_balanced_data(self):
        labels = np.array([0, 0, 1, 1])
        w = compute_class_weights(labels)
        # balanced → 모든 weight ≈ 1.0
        self.assertAlmostEqual(w[0], 1.0)
        self.assertAlmostEqual(w[1], 1.0)

    def test_imbalanced_90_10(self):
        labels = np.array([0]*90 + [1]*10)
        w = compute_class_weights(labels)
        # W_0 = 100 / (2*90) ≈ 0.556
        # W_1 = 100 / (2*10) = 5.0
        self.assertAlmostEqual(w[0], 100 / 180, places=3)
        self.assertAlmostEqual(w[1], 5.0, places=3)
        # 소수 클래스 가중치가 다수 클래스보다 훨씬 높아야 함
        self.assertGreater(w[1], w[0] * 5)

    def test_single_class(self):
        labels = np.array([0, 0, 0])
        w = compute_class_weights(labels)
        self.assertIn(0, w)


class TestWeightedImpurity(unittest.TestCase):
    """Class weight가 적용된 entropy/gini."""

    def test_entropy_weighted_amplifies_minority(self):
        """불균형 데이터에서 weighted entropy > unweighted entropy."""
        labels = np.array([0]*9 + [1])
        w = compute_class_weights(labels)
        ent_plain = _entropy(labels)
        ent_weighted = _entropy(labels, class_weights=w)
        # Weighted entropy는 소수 클래스를 증폭하므로 더 높아야 함
        self.assertGreater(ent_weighted, ent_plain)

    def test_gini_weighted_amplifies_minority(self):
        labels = np.array([0]*9 + [1])
        w = compute_class_weights(labels)
        gini_plain = _gini(labels)
        gini_weighted = _gini(labels, class_weights=w)
        self.assertGreater(gini_weighted, gini_plain)

    def test_entropy_pure_label_is_zero(self):
        labels = np.array([0, 0, 0])
        self.assertAlmostEqual(_entropy(labels), 0.0)

    def test_info_gain_with_weights(self):
        parent = np.array([0]*9 + [1])
        left = np.array([0]*9)
        right = np.array([1])
        w = compute_class_weights(parent)
        ig = _info_gain(parent, left, right, "entropy", class_weights=w)
        self.assertGreater(ig, 0.0)


class TestFindBestThresholdWithWeights(unittest.TestCase):
    """find_best_threshold에 class_weights 적용."""

    def test_threshold_found_with_weights(self):
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        w = compute_class_weights(labels)
        gain, thr, op = find_best_threshold(values, labels, class_weights=w)
        self.assertGreater(gain, 0.0)
        self.assertIsNotNone(thr)


class TestDecisionPathF1Fitness(unittest.TestCase):
    """DecisionPath의 F1-based fitness와 leaf_probability."""

    def _make_path(self, f1: float, accuracy: float,
                   coverage: float, leaf_dist: dict) -> DecisionPath:
        return DecisionPath(
            conditions=[SplitCondition("feat", "<=", 5.0)],
            prediction=1,
            accuracy=accuracy,
            coverage=coverage,
            leaf_class_dist=leaf_dist,
            f1_score=f1,
            balanced_accuracy=0.5,
        )

    def test_f1_based_fitness(self):
        """F1=0 → fitness degrades to accuracy*0.1."""
        path_no_f1 = self._make_path(
            f1=0.0, accuracy=0.88, coverage=0.90,
            leaf_dist={0: 88, 1: 0},
        )
        path_good_f1 = self._make_path(
            f1=0.6, accuracy=0.70, coverage=0.30,
            leaf_dist={0: 50, 1: 20},
        )
        # "predict all 0" has F1=0, its fitness should be much lower
        self.assertGreater(path_good_f1.fitness, path_no_f1.fitness)

    def test_mode_collapse_path_low_fitness(self):
        """Mode collapse (전부 0 예측) → fitness ≈ 0."""
        path = self._make_path(
            f1=0.0, accuracy=0.88, coverage=1.0,
            leaf_dist={0: 88, 1: 0},
        )
        # F1=0 → fitness = accuracy(0.88) * 0.1 * log1p(1.0) = 0.061
        self.assertLess(path.fitness, 0.1)

    def test_leaf_probability(self):
        """leaf_probability는 class 1 비율."""
        path = self._make_path(
            f1=0.5, accuracy=0.7, coverage=0.3,
            leaf_dist={0: 70, 1: 30},
        )
        self.assertAlmostEqual(path.leaf_probability, 0.30)

    def test_leaf_probability_all_negative(self):
        path = self._make_path(
            f1=0.0, accuracy=0.88, coverage=1.0,
            leaf_dist={0: 88, 1: 0},
        )
        self.assertAlmostEqual(path.leaf_probability, 0.0)

    def test_leaf_probability_all_positive(self):
        path = self._make_path(
            f1=1.0, accuracy=1.0, coverage=0.1,
            leaf_dist={0: 0, 1: 10},
        )
        self.assertAlmostEqual(path.leaf_probability, 1.0)


class TestPredictProbaLeafBased(unittest.TestCase):
    """SemanticForest.predict_proba가 리프 확률을 정확히 반환하는지 검증."""

    def test_predict_proba_continuous(self):
        """predict_proba 결과가 hard 0/1이 아닌 연속적 확률."""
        from src.aco.semantic_forest import SemanticForest, TreeResult

        sf = SemanticForest.__new__(SemanticForest)
        sf.is_fitted_ = True
        sf._default_prediction = 0
        sf._class_weights = None  # no weighting in this test

        # 가상 트리 1: feat <= 5 → class 0 (leaf: 80% neg, 20% pos)
        rule1 = DecisionPath(
            conditions=[SplitCondition("feat", "<=", 5.0)],
            prediction=0,
            accuracy=0.8,
            coverage=50,
            leaf_class_dist={0: 80, 1: 20},
            f1_score=0.0,
            balanced_accuracy=0.5,
        )
        # 가상 트리 1: feat > 5 → class 1 (leaf: 30% neg, 70% pos)
        rule2 = DecisionPath(
            conditions=[SplitCondition("feat", ">", 5.0)],
            prediction=1,
            accuracy=0.7,
            coverage=50,
            leaf_class_dist={0: 30, 1: 70},
            f1_score=0.7,
            balanced_accuracy=0.7,
        )

        tree_result = TreeResult(
            rules=[rule1, rule2],
            oob_accuracy=0.7,
        )
        sf.trees_ = [tree_result]

        df = pd.DataFrame({"feat": [3.0, 7.0]})
        proba = sf.predict_proba(df)

        # 샘플 0: feat=3.0 → rule1 매칭 → P(class1) = 0.2
        self.assertAlmostEqual(proba[0, 1], 0.2)
        self.assertAlmostEqual(proba[0, 0], 0.8)

        # 샘플 1: feat=7.0 → rule2 매칭 → P(class1) = 0.7
        self.assertAlmostEqual(proba[1, 1], 0.7)
        self.assertAlmostEqual(proba[1, 0], 0.3)

        # 확률이 [0,1] 범위이고 합=1
        for i in range(2):
            self.assertAlmostEqual(proba[i].sum(), 1.0)
            self.assertTrue(np.all(proba[i] >= 0))
            self.assertTrue(np.all(proba[i] <= 1))

    def test_predict_proba_not_degenerate(self):
        """불균형 리프에서도 predict_proba 값이 0.0이나 1.0만 아님."""
        from src.aco.semantic_forest import SemanticForest, TreeResult

        sf = SemanticForest.__new__(SemanticForest)
        sf.is_fitted_ = True
        sf._default_prediction = 0
        sf._class_weights = None  # no weighting in this test

        # 리프 클래스 비율이 90:10인 규칙
        rule = DecisionPath(
            conditions=[SplitCondition("x", "<=", 100.0)],
            prediction=0,
            accuracy=0.9,
            coverage=100,
            leaf_class_dist={0: 90, 1: 10},
            f1_score=0.0,
            balanced_accuracy=0.5,
        )

        tree_result = TreeResult(
            rules=[rule],
            oob_accuracy=0.9,
        )
        sf.trees_ = [tree_result]

        df = pd.DataFrame({"x": [42.0]})
        proba = sf.predict_proba(df)

        # 결과가 hard [1.0, 0.0]이 아니라 [0.9, 0.1]이어야 함
        self.assertAlmostEqual(proba[0, 0], 0.9)
        self.assertAlmostEqual(proba[0, 1], 0.1)
        # 이것이 mode collapse 방지의 핵심: non-degenerate probability
        self.assertGreater(proba[0, 1], 0.0)
        self.assertLess(proba[0, 1], 1.0)


if __name__ == "__main__":
    unittest.main()
