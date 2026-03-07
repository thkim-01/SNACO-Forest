from typing import Dict, List, Optional, Tuple
import os
import numpy as np
from src.sdt.tree import TreeNode, SemanticDecisionTree
from src.sdt.logic_refinement import OntologyRefinementGenerator, load_refinements_json
from src.utils.compute_backend import calculate_entropy, calculate_gini, resolve_backend, resolve_torch_device


class LogicTreeNode(TreeNode):
    """TreeNode subclass for Logic SDT that handles Ontology instances."""
    def __init__(self, instances: List, depth: int = 0, node_id: int = 0):
        self.instances = instances
        self.depth = depth
        self.node_id = node_id
        self.refinement = None
        self.is_leaf = False
        self.predicted_label = None
        self.left_child = None
        self.right_child = None
        
        self.num_instances = len(instances)
        self.label_counts = self._count_labels()
        self.gini = self._calculate_gini()

    def _get_label_from_inst(self, inst):
        if hasattr(inst, 'hasLabel'):
            return inst.hasLabel[0] if inst.hasLabel else None
        return getattr(inst, 'label', None)

    def _count_labels(self) -> dict:
        counts = {}
        for inst in self.instances:
            label = self._get_label_from_inst(inst)
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
        return counts
    
    def _calculate_gini(self) -> float:
        if self.num_instances == 0:
            return 0.0
        gini = 1.0
        total = sum(self.label_counts.values())
        if total == 0:
            return 0.0
        for count in self.label_counts.values():
            if count > 0:
                p = count / total
                gini -= p * p
        return gini


class LogicSDTLearner:
    """Semantic Decision Tree Learner using Ontology/Logic-based Refinements."""
    
    def __init__(
        self,
        ontology_manager,
        max_depth: int = 10,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = None,
        verbose: bool = True,
        refinement_mode: str = 'dynamic',
        refinement_file: Optional[str] = None,
        split_criterion: str = 'information_gain',
        search_strategy: str = 'exhaustive',
        compute_backend: str = 'auto',
        torch_device: str = 'auto',
        random_state: int = 42,
        heuristic_probe_samples: int = 128,
        aco_num_ants: int = 12,
        aco_num_iterations: int = 15,
        aco_alpha: float = 1.0,
        aco_beta: float = 2.0,
        aco_evaporation_rate: float = 0.25,
        aco_q: float = 1.0,
        aco_explore_prob: float = 0.1,
        dataset_name: Optional[str] = None,
        dataset_concepts: Optional[dict] = None,
        dataset_refinement_profile: Optional[dict] = None,
        enable_toxicity_seed_rules: bool = False,
        reasoner_engine: str = 'none',
        reasoner_infer_property_values: bool = False,
        pig_alpha: float = 1.0,
        semantic_weight: float = 0.3,
    ):
        self.onto_manager = ontology_manager
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.verbose = verbose
        self.refinement_mode = refinement_mode
        self.refinement_file = refinement_file
        self.split_criterion = split_criterion
        self.search_strategy = str(search_strategy or 'exhaustive').strip().lower()
        self.compute_backend = resolve_backend(compute_backend)
        self.torch_device = resolve_torch_device(torch_device)
        self.random_state = int(random_state)
        self.heuristic_probe_samples = max(8, int(heuristic_probe_samples))
        self.aco_num_ants = max(1, int(aco_num_ants))
        self.aco_num_iterations = max(1, int(aco_num_iterations))
        self.aco_alpha = max(0.0, float(aco_alpha))
        self.aco_beta = max(0.0, float(aco_beta))
        self.aco_evaporation_rate = min(0.95, max(0.0, float(aco_evaporation_rate)))
        self.aco_q = max(0.0, float(aco_q))
        self.aco_explore_prob = min(1.0, max(0.0, float(aco_explore_prob)))
        self._rng = np.random.default_rng(self.random_state)
        self.dataset_name = dataset_name
        self.dataset_concepts = dataset_concepts or {}
        self.dataset_refinement_profile = dataset_refinement_profile or {}
        self.enable_toxicity_seed_rules = enable_toxicity_seed_rules
        self.reasoner_engine = reasoner_engine
        self.reasoner_infer_property_values = reasoner_infer_property_values
        self.pig_alpha = float(pig_alpha)
        self.semantic_weight = float(semantic_weight)

        # TaxonomyScorer: PIG / SemanticSimilarity 분할 기준 지원
        self._taxonomy_scorer = None
        if split_criterion in ('pig', 'semantic_similarity', 'pig_semantic'):
            try:
                from src.sdt.taxonomy_scorer import TaxonomyScorer
                onto = getattr(ontology_manager, 'onto', None)
                self._taxonomy_scorer = TaxonomyScorer(
                    ontology=onto,
                    alpha=self.pig_alpha,
                )
                if self.verbose:
                    print(f"[TaxonomyScorer] Initialized (alpha={self.pig_alpha}, "
                          f"semantic_weight={self.semantic_weight})")
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] TaxonomyScorer init failed: {e}. "
                          "Falling back to entropy-based IG.")


        static_refs = None
        if refinement_mode == 'static':
            if not refinement_file:
                raise ValueError("refinement_file is required when refinement_mode='static'")
            if not os.path.exists(refinement_file):
                raise FileNotFoundError(refinement_file)
            static_refs = load_refinements_json(ontology_manager.onto, refinement_file)

        self.refinement_generator = OntologyRefinementGenerator(
            ontology_manager,
            static_refinements=static_refs,
            dataset_name=self.dataset_name,
            dataset_concepts=self.dataset_concepts,
            dataset_refinement_profile=self.dataset_refinement_profile,
            enable_toxicity_seed_rules=self.enable_toxicity_seed_rules,
            reasoner_engine=self.reasoner_engine,
            reasoner_infer_property_values=self.reasoner_infer_property_values,
        )
        self.tree = None
        self.class_weights_dict = {}

        if self.verbose and self.compute_backend == 'torch':
            print(f"[ComputeBackend] torch enabled (device={self.torch_device})")
        if self.verbose and self.search_strategy in ('aco', 'ant_colony'):
            print(
                "[Search] ACO enabled "
                f"(ants={self.aco_num_ants}, iterations={self.aco_num_iterations}, "
                f"alpha={self.aco_alpha}, beta={self.aco_beta})"
            )

    def fit(self, instances: List):
        """Train the SDT"""
        self.tree = SemanticDecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )
        
        if self.class_weight == 'balanced':
            self.class_weights_dict = self._compute_class_weights(instances)
            if self.verbose:
                print(f"Class weights: {self.class_weights_dict}")
        
        root_node = LogicTreeNode(instances, depth=0, node_id=self.tree.get_next_node_id())
        self.tree.root = root_node
        self.tree.nodes.append(root_node)
        
        self._build_tree(root_node, center_class=self.onto_manager.Molecule)
        
        if self.verbose:
            total_nodes = len(self.tree.nodes)
            print(f"Logic SDT training completed. Total nodes: {total_nodes}")
            
        return self.tree

    def _build_tree(self, node: TreeNode, center_class):
        """Recursive tree building"""
        unique_labels = {self._get_label(inst) for inst in node.instances}
        if (
            node.depth >= self.max_depth
            or node.num_instances < self.min_samples_split
            or len(unique_labels) == 1
        ):
            self._set_leaf(node)
            return

        candidate_refinements = self.refinement_generator.generate_refinements(
            center_class,
            node.instances,
        )

        if self.verbose and node.depth == 0:
            print(f"[DEBUG] Root node: {node.num_instances} instances, "
                  f"unique_labels={unique_labels}, "
                  f"candidate_refinements={len(candidate_refinements) if candidate_refinements else 0}")
            if candidate_refinements:
                print(f"[DEBUG] First 5 refinements: {candidate_refinements[:5]}")

        if not candidate_refinements:
            self._set_leaf(node)
            return

        if self.search_strategy in ('aco', 'ant_colony'):
            best_refinement, best_gain, best_left_instances, best_right_instances = self._find_best_refinement_aco(
                node,
                candidate_refinements,
            )
        else:
            best_refinement, best_gain, best_left_instances, best_right_instances = self._find_best_refinement_exhaustive(
                node,
                candidate_refinements,
            )

        if best_refinement is None:
            self._set_leaf(node)
            return

        node.refinement = best_refinement
        node.is_leaf = False

        left_node = LogicTreeNode(
            best_left_instances,
            depth=node.depth + 1,
            node_id=self.tree.get_next_node_id(),
        )
        right_node = LogicTreeNode(
            best_right_instances,
            depth=node.depth + 1,
            node_id=self.tree.get_next_node_id(),
        )

        node.left_child = left_node
        node.right_child = right_node

        self.tree.nodes.append(left_node)
        self.tree.nodes.append(right_node)

        self._build_tree(left_node, center_class=center_class)
        self._build_tree(right_node, center_class=center_class)

    def _set_leaf(self, node: TreeNode):
        """Mark node as leaf and set predicted label"""
        node.is_leaf = True
        labels = list(node.label_counts.items())
        if labels:
            node.predicted_label = max(labels, key=lambda x: x[1])[0]
        else:
            node.predicted_label = None

    def predict(self, instances: List):
        """Predict labels for instances"""
        if self.tree is None or self.tree.root is None:
            raise ValueError("Model not fitted yet")
        
        predictions = []
        for inst in instances:
            node = self.tree.root
            while not node.is_leaf:
                try:
                    if self.refinement_generator.instance_satisfies_refinement(inst, node.refinement):
                        node = node.left_child
                    else:
                        node = node.right_child
                    if node is None:
                        break
                except Exception:
                    node = node.right_child
                    if node is None:
                        break
            
            if node and node.is_leaf:
                predictions.append(node.predicted_label)
            else:
                predictions.append(None)
        
        return predictions

    def _calculate_information_gain(self, parent_instances, left_instances, right_instances,
                                    refinement=None):
        """Calculate information gain for a split.

        Supports criteria: information_gain, gini, pig, semantic_similarity, pig_semantic.
        """
        if self.split_criterion == 'gini':
            return self._calculate_gini_gain(parent_instances, left_instances, right_instances)
        elif self.split_criterion == 'pig':
            return self._calculate_pig_gain(
                parent_instances, left_instances, right_instances, refinement
            )
        elif self.split_criterion == 'semantic_similarity':
            return self._calculate_semantic_sim_gain(
                parent_instances, left_instances, right_instances, refinement
            )
        elif self.split_criterion == 'pig_semantic':
            return self._calculate_pig_semantic_gain(
                parent_instances, left_instances, right_instances, refinement
            )
        else:
            return self._calculate_entropy_gain(parent_instances, left_instances, right_instances)

    def _evaluate_refinement_split(self, parent_instances: List, refinement) -> Optional[Tuple[float, List, List]]:
        left_instances = []
        right_instances = []

        for inst in parent_instances:
            try:
                if self.refinement_generator.instance_satisfies_refinement(inst, refinement):
                    left_instances.append(inst)
                else:
                    right_instances.append(inst)
            except Exception:
                right_instances.append(inst)

        if len(left_instances) < self.min_samples_leaf or len(right_instances) < self.min_samples_leaf:
            return None

        gain = self._calculate_information_gain(
            parent_instances, left_instances, right_instances, refinement=refinement
        )
        return gain, left_instances, right_instances

    def _find_best_refinement_exhaustive(self, node: TreeNode, candidate_refinements: List):
        best_refinement = None
        best_gain = -1.0
        best_left_instances = None
        best_right_instances = None

        for refinement in candidate_refinements:
            evaluated = self._evaluate_refinement_split(node.instances, refinement)
            if evaluated is None:
                continue
            gain, left_instances, right_instances = evaluated

            if node.depth == 0:
                print(
                    f"[DEBUG] Refinement {refinement}: left={len(left_instances)}, "
                    f"right={len(right_instances)}, gain={gain:.4f}"
                )

            if gain > best_gain:
                best_gain = gain
                best_refinement = refinement
                best_left_instances = left_instances
                best_right_instances = right_instances

        return best_refinement, best_gain, best_left_instances, best_right_instances

    def _estimate_refinement_heuristic(self, instances: List, refinement) -> float:
        total = len(instances)
        if total == 0:
            return 0.0

        probe_size = min(total, self.heuristic_probe_samples)
        if probe_size < total:
            sampled_idx = self._rng.choice(total, size=probe_size, replace=False)
            sample_instances = [instances[int(i)] for i in sampled_idx]
        else:
            sample_instances = instances

        left_labels = []
        right_labels = []
        for inst in sample_instances:
            try:
                sat = self.refinement_generator.instance_satisfies_refinement(inst, refinement)
            except Exception:
                sat = False
            label = self._get_label(inst)
            if sat:
                left_labels.append(label)
            else:
                right_labels.append(label)

        left_n = len(left_labels)
        right_n = len(right_labels)
        if left_n == 0 or right_n == 0:
            return 1e-9

        balance = min(left_n, right_n) / max(left_n, right_n)

        def _impurity(labels: List):
            n = len(labels)
            if n == 0:
                return 0.0
            counts = {}
            for y in labels:
                counts[y] = counts.get(y, 0) + 1
            imp = 1.0
            for c in counts.values():
                p = c / n
                imp -= p * p
            return imp

        parent_imp = _impurity(left_labels + right_labels)
        child_imp = (left_n / probe_size) * _impurity(left_labels) + (right_n / probe_size) * _impurity(right_labels)
        approx_gain = max(0.0, parent_imp - child_imp)
        return max(1e-9, approx_gain * (0.5 + 0.5 * balance))

    def _find_best_refinement_aco(self, node: TreeNode, candidate_refinements: List):
        n_candidates = len(candidate_refinements)
        if n_candidates == 0:
            return None, -1.0, None, None

        pheromone = np.ones(n_candidates, dtype=float)
        heuristic = np.array(
            [self._estimate_refinement_heuristic(node.instances, ref) for ref in candidate_refinements],
            dtype=float,
        )
        heuristic = np.maximum(heuristic, 1e-9)

        eval_cache: Dict[int, Optional[Tuple[float, List, List]]] = {}
        best_refinement = None
        best_gain = -1.0
        best_left_instances = None
        best_right_instances = None

        for _ in range(self.aco_num_iterations):
            for _ in range(self.aco_num_ants):
                if self._rng.random() < self.aco_explore_prob:
                    idx = int(self._rng.integers(0, n_candidates))
                else:
                    desirability = np.power(np.maximum(pheromone, 1e-12), self.aco_alpha) * np.power(heuristic, self.aco_beta)
                    denom = float(np.sum(desirability))
                    if denom <= 0:
                        idx = int(self._rng.integers(0, n_candidates))
                    else:
                        probs = desirability / denom
                        idx = int(self._rng.choice(n_candidates, p=probs))

                if idx not in eval_cache:
                    eval_cache[idx] = self._evaluate_refinement_split(node.instances, candidate_refinements[idx])

                evaluated = eval_cache[idx]
                if evaluated is None:
                    continue

                gain, left_instances, right_instances = evaluated
                if gain > best_gain:
                    best_gain = gain
                    best_refinement = candidate_refinements[idx]
                    best_left_instances = left_instances
                    best_right_instances = right_instances

                pheromone[idx] += self.aco_q * max(0.0, gain)

            pheromone *= (1.0 - self.aco_evaporation_rate)
            pheromone = np.maximum(pheromone, 1e-12)

        if best_refinement is None:
            ranked = np.argsort(pheromone)[::-1]
            for idx in ranked[: min(10, n_candidates)]:
                if int(idx) not in eval_cache:
                    eval_cache[int(idx)] = self._evaluate_refinement_split(
                        node.instances,
                        candidate_refinements[int(idx)],
                    )
                evaluated = eval_cache[int(idx)]
                if evaluated is None:
                    continue
                gain, left_instances, right_instances = evaluated
                if gain > best_gain:
                    best_gain = gain
                    best_refinement = candidate_refinements[int(idx)]
                    best_left_instances = left_instances
                    best_right_instances = right_instances

        return best_refinement, best_gain, best_left_instances, best_right_instances

    def _calculate_entropy_gain(self, parent, left, right):
        """Calculate entropy-based information gain"""
        n = len(parent)
        if n == 0:
            return 0.0
        
        parent_entropy = calculate_entropy(
            parent, 
            self._get_label, 
            self.class_weights_dict, 
            self.compute_backend, 
            self.torch_device
        )
        left_entropy = calculate_entropy(
            left, 
            self._get_label, 
            self.class_weights_dict, 
            self.compute_backend, 
            self.torch_device
        )
        right_entropy = calculate_entropy(
            right, 
            self._get_label, 
            self.class_weights_dict, 
            self.compute_backend, 
            self.torch_device
        )
        
        weighted_entropy = (len(left) / n) * left_entropy + (len(right) / n) * right_entropy
        return parent_entropy - weighted_entropy

    def _calculate_gini_gain(self, parent, left, right):
        """Calculate gini-based information gain"""
        n = len(parent)
        if n == 0:
            return 0.0
        
        parent_gini = calculate_gini(
            parent, 
            self._get_label, 
            self.class_weights_dict, 
            self.compute_backend, 
            self.torch_device
        )
        left_gini = calculate_gini(
            left, 
            self._get_label, 
            self.class_weights_dict, 
            self.compute_backend, 
            self.torch_device
        )
        right_gini = calculate_gini(
            right, 
            self._get_label, 
            self.class_weights_dict, 
            self.compute_backend, 
            self.torch_device
        )
        
        weighted_gini = (len(left) / n) * left_gini + (len(right) / n) * right_gini
        return parent_gini - weighted_gini

    def _compute_class_weights(self, instances):
        """Compute class weights for imbalanced data"""
        label_counts = {}
        for inst in instances:
            label = self._get_label(inst)
            if label is not None:
                label_counts[label] = label_counts.get(label, 0) + 1
        
        total = sum(label_counts.values())
        weights = {}
        for label, count in label_counts.items():
            weights[label] = total / (len(label_counts) * count) if count > 0 else 1.0
        
        return weights

    def _get_label(self, inst):
        """Extract label from instance"""
        if hasattr(inst, 'hasLabel'):
            return inst.hasLabel[0] if inst.hasLabel else None
        return getattr(inst, 'label', None)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PIG (Penalized Information Gain) 분할 기준
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _get_instance_concepts(self, inst) -> list:
        """인스턴스에서 관련 온톨로지 개념(클래스)을 추출한다."""
        concepts = []
        for t in getattr(inst, 'is_a', []):
            if hasattr(t, 'name') and hasattr(t, 'ancestors'):
                concepts.append(t)
        return concepts

    def _calculate_pig_gain(self, parent, left, right, refinement=None):
        """Penalized Information Gain.

        PIG = IG × (1 + log(1 + α × ATI))

        IG는 기본 entropy-based information gain.
        ATI는 refinement가 참조하는 개념들의 평균 Taxonomic Informativeness.
        """
        import math

        ig = self._calculate_entropy_gain(parent, left, right)

        if self._taxonomy_scorer is None or refinement is None:
            return ig

        ati = self._taxonomy_scorer.compute_ati_for_refinement(refinement)
        pf = math.log(1.0 + self.pig_alpha * ati)
        return ig * (1.0 + pf)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Semantic Similarity 기반 분할 기준
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _calculate_semantic_sim_gain(self, parent, left, right, refinement=None):
        """Semantic Similarity 기반 분할 점수.

        Score = (1 - w) × IG + w × Sim(A) × IG_scale

        Sim(A) = Σ_u p_u × Sim(a_u)
        각 부분집합의 intra-similarity를 크기 비율로 가중 합산한다.
        """
        ig = self._calculate_entropy_gain(parent, left, right)

        if self._taxonomy_scorer is None:
            return ig

        sim_score = self._taxonomy_scorer.semantic_similarity_split_score(
            left, right, self._get_instance_concepts
        )

        # IG 스케일에 맞추어 결합
        score = ((1.0 - self.semantic_weight) * ig
                 + self.semantic_weight * sim_score * max(ig, 0.001))
        return score

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PIG + Semantic Similarity 결합 분할 기준
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _calculate_pig_semantic_gain(self, parent, left, right, refinement=None):
        """PIG + Semantic Similarity 결합 점수.

        Score = (1 - w) × PIG + w × Sim(A) × IG_scale
        """
        pig = self._calculate_pig_gain(parent, left, right, refinement)

        if self._taxonomy_scorer is None:
            return pig

        ig = self._calculate_entropy_gain(parent, left, right)
        sim_score = self._taxonomy_scorer.semantic_similarity_split_score(
            left, right, self._get_instance_concepts
        )

        score = ((1.0 - self.semantic_weight) * pig
                 + self.semantic_weight * sim_score * max(ig, 0.001))
        return score
