"""
Unit Tests — 6대 독성 데이터셋 파이프라인 모듈

테스트 대상:
    1. OntologyRouter: config 로드, route(), 캐시, 플레이스홀더
    2. KnowledgeHeuristics: PriorBounds, StaticPheromone, BindingPocket
    3. HierarchicalSearch: 게이트 설치, 후보 필터링, 엔트로피
    4. DataPipeline: load_dataset, stratified_split, compute_metrics, validate_graph
"""

import sys
import os
import json
import math
import unittest
import tempfile
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import networkx as nx

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 유틸리티: 테스트용 미니 그래프 & config
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_mini_graph(n_nodes: int = 20) -> nx.DiGraph:
    """테스트용 소형 온톨로지 그래프 생성."""
    G = nx.DiGraph()
    G.add_node("Thing", label="Thing", uri="urn:Thing")
    for i in range(n_nodes):
        nid = f"node_{i}"
        G.add_node(nid, label=f"concept_{i}", uri=f"urn:test:{i}")
        G.add_edge("Thing", nid, predicate="subClassOf", pheromone=1.0)
    # 체인 구조: node_0 → node_1 → node_2 → ...
    for i in range(n_nodes - 1):
        G.add_edge(f"node_{i}", f"node_{i+1}",
                    predicate="relatedTo", pheromone=1.0)
    return G


def _make_feature_graph() -> nx.DiGraph:
    """feature 노드가 있는 테스트 그래프."""
    G = _make_mini_graph(10)
    features = ["molecular_weight", "logp", "tpsa", "num_hba",
                 "num_hbd", "num_aromatic_rings", "num_nitrogens",
                 "num_oxygens", "num_halogens", "num_rings"]
    for feat in features:
        fid = f"feature:{feat}"
        G.add_node(fid, label=feat, uri=f"urn:feature:{feat}",
                    is_feature=True, feature_key=feat)
        G.add_edge("Thing", fid, predicate="hasFeature", pheromone=1.0)
    return G


def _make_feature_df(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """테스트용 분자 디스크립터 DataFrame."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "molecular_weight": rng.normal(350, 100, n_samples).clip(50, 900),
        "logp": rng.normal(2.5, 1.5, n_samples),
        "tpsa": rng.normal(75, 30, n_samples).clip(0, 200),
        "num_hba": rng.poisson(4, n_samples).astype(float),
        "num_hbd": rng.poisson(1.5, n_samples).astype(float),
        "num_aromatic_rings": rng.poisson(2, n_samples).astype(float),
        "num_nitrogens": rng.poisson(2, n_samples).astype(float),
        "num_oxygens": rng.poisson(3, n_samples).astype(float),
        "num_halogens": rng.poisson(0.5, n_samples).astype(float),
        "num_rings": rng.poisson(3, n_samples).astype(float),
    })


def _make_labels(n_samples: int = 100, seed: int = 42) -> np.ndarray:
    """이진 라벨."""
    rng = np.random.RandomState(seed)
    return (rng.random(n_samples) > 0.5).astype(int)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. OntologyRouter 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestOntologyRouter(unittest.TestCase):
    """OntologyRouter: config 로드, 라우팅, 캐시, 플레이스홀더."""

    @classmethod
    def setUpClass(cls):
        from src.aco.ontology_router import OntologyRouter, _clear_cache
        _clear_cache()
        cls.config_path = Path(ROOT) / "configs" / "dataset_ontology_config.json"
        cls.base_dir = Path(ROOT)
        if cls.config_path.exists():
            cls.router = OntologyRouter(
                config_path=str(cls.config_path),
                base_dir=str(cls.base_dir),
                use_cache=False,
            )
        else:
            cls.router = None

    def test_config_loaded(self):
        """config.json이 정상 로드되고 6개 데이터셋이 등록됨."""
        if self.router is None:
            self.skipTest("config not found")
        self.assertEqual(len(self.router.available_datasets), 6)
        expected = {"bace", "bbbp", "clintox", "hiv", "sider", "tox21"}
        self.assertEqual(set(self.router.available_datasets), expected)

    def test_ontology_registry(self):
        """5개 온톨로지(dto, chebi, go, mesh, bao) 등록됨."""
        if self.router is None:
            self.skipTest("config not found")
        expected = {"chebi", "dto", "go", "mesh", "bao"}
        self.assertEqual(set(self.router.available_ontologies), expected)

    def test_get_required_ontologies_bbbp(self):
        """BBBP -> chebi (1:1)."""
        if self.router is None:
            self.skipTest("config not found")
        required = self.router.get_required_ontologies("bbbp")
        self.assertEqual(required, ["chebi"])

    def test_get_required_ontologies_hiv(self):
        """HIV -> bao (1:1)."""
        if self.router is None:
            self.skipTest("config not found")
        required = self.router.get_required_ontologies("hiv")
        self.assertEqual(required, ["bao"])

    def test_get_required_ontologies_sider(self):
        """SIDER -> mesh (1:1)."""
        if self.router is None:
            self.skipTest("config not found")
        required = self.router.get_required_ontologies("sider")
        self.assertEqual(required, ["mesh"])

    def test_get_required_ontologies_tox21(self):
        """Tox21 -> go (1:1)."""
        if self.router is None:
            self.skipTest("config not found")
        required = self.router.get_required_ontologies("tox21")
        self.assertEqual(required, ["go"])

    def test_bridge_domain_mapping(self):
        """bridge_domain: chebi for BBBP/ClinTox, anchor for others."""
        if self.router is None:
            self.skipTest("config not found")
        self.assertEqual(self.router.get_bridge_domain("bbbp"), "chebi")
        self.assertEqual(self.router.get_bridge_domain("clintox"), "chebi")
        self.assertEqual(self.router.get_bridge_domain("hiv"), "anchor")
        self.assertEqual(self.router.get_bridge_domain("bace"), "anchor")
        self.assertEqual(self.router.get_bridge_domain("sider"), "anchor")
        self.assertEqual(self.router.get_bridge_domain("tox21"), "anchor")

    def test_unknown_dataset_raises(self):
        """존재하지 않는 데이터셋 이름은 KeyError."""
        if self.router is None:
            self.skipTest("config not found")
        with self.assertRaises(KeyError):
            self.router.get_dataset_config("nonexistent_dataset")

    def test_get_dataset_config_bace(self):
        """BACE config에 pocket_features가 포함됨."""
        if self.router is None:
            self.skipTest("config not found")
        cfg = self.router.get_dataset_config("bace")
        self.assertEqual(cfg["heuristic_profile"], "binding_pocket")
        self.assertIn("pocket_features", cfg)
        self.assertIn("s1_occupancy", cfg["pocket_features"])

    def test_get_dataset_config_clintox(self):
        """ClinTox config에 static_pheromone이 포함됨."""
        if self.router is None:
            self.skipTest("config not found")
        cfg = self.router.get_dataset_config("clintox")
        self.assertEqual(cfg["heuristic_profile"], "fda_cramer_static")
        sp = cfg.get("static_pheromone", {})
        self.assertEqual(len(sp.get("fda_edt_questions", [])), 6)

    def test_defaults(self):
        """기본 SemanticForest 파라미터가 올바름."""
        if self.router is None:
            self.skipTest("config not found")
        defaults = self.router.get_defaults()
        self.assertEqual(defaults["n_trees"], 8)
        self.assertEqual(defaults["n_ants_per_tree"], 25)
        self.assertEqual(defaults["n_generations"], 3)
        self.assertAlmostEqual(defaults["evaporation_rate"], 0.10)

    def test_route_dto_only(self):
        """DTO only 데이터셋(bace) 라우팅으로 그래프 반환."""
        if self.router is None:
            self.skipTest("config not found")
        dto_path = self.base_dir / "ontology" / "DTO.xrdf"
        if not dto_path.exists():
            self.skipTest("DTO.xrdf not found")
        G = self.router.route("bace")
        self.assertIsInstance(G, nx.DiGraph)
        self.assertGreater(G.number_of_nodes(), 100)
        self.assertGreater(G.number_of_edges(), 100)

    def test_placeholder_ontology(self):
        """플레이스홀더 온톨로지가 올바르게 생성됨."""
        from src.aco.ontology_router import OntologyRouter
        router = OntologyRouter.__new__(OntologyRouter)
        # _create_placeholder_ontology 직접 호출
        G = router._create_placeholder_ontology("chebi", ["CHEBI:24431"])
        self.assertIn("chebi:Thing", G)
        self.assertIn("chebi:chemical_entity", G)
        self.assertIn("chebi:organic_molecule", G)
        self.assertGreater(G.number_of_nodes(), 5)

        G_go = router._create_placeholder_ontology("go", ["GO:0008150"])
        self.assertIn("go:biological_process", G_go)
        self.assertIn("go:viral_process", G_go)

        G_mesh = router._create_placeholder_ontology("mesh", [])
        self.assertIn("mesh:system_organ_class", G_mesh)
        self.assertIn("mesh:nervous_system_disorders", G_mesh)

    def test_cache_behavior(self):
        """캐시 활성화 시 같은 온톨로지를 두 번 로드해도 파싱은 1번."""
        from src.aco.ontology_router import _clear_cache, _ONTOLOGY_CACHE
        _clear_cache()
        self.assertEqual(len(_ONTOLOGY_CACHE), 0)

        if self.router is None:
            self.skipTest("config not found")
        dto_path = self.base_dir / "ontology" / "DTO.xrdf"
        if not dto_path.exists():
            self.skipTest("DTO.xrdf not found")

        # 캐시 활성 라우터
        from src.aco.ontology_router import OntologyRouter
        cached_router = OntologyRouter(
            config_path=str(self.config_path),
            base_dir=str(self.base_dir),
            use_cache=True,
        )
        G1 = cached_router.route("bace")
        self.assertIn("dto", _ONTOLOGY_CACHE)
        G2 = cached_router.route("bace")  # DTO reuse from cache
        # bace uses dto, which was cached from first route("bace")
        self.assertIn("dto", _ONTOLOGY_CACHE)
        _clear_cache()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. KnowledgeHeuristics 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPhysicochemicalPriorBounds(unittest.TestCase):
    """PhysicochemicalPriorBounds: BBBP 물리화학적 Prior 범위 제한."""

    def setUp(self):
        from src.aco.knowledge_heuristics import PhysicochemicalPriorBounds
        self.G = _make_feature_graph()
        self.bounds = {
            "tpsa": {"low": 0, "high": 90},
            "logp": {"low": -0.5, "high": 5.0},
            "molecular_weight": {"low": 150, "high": 500},
        }
        self.ppb = PhysicochemicalPriorBounds(self.bounds, self.G)

    def test_apply_clips_values(self):
        """DataFrame 값이 prior bounds로 클리핑됨."""
        df = _make_feature_df(50)
        clipped = self.ppb.apply(df)
        self.assertTrue(clipped["tpsa"].max() <= 90.0)
        self.assertTrue(clipped["tpsa"].min() >= 0.0)
        self.assertTrue(clipped["logp"].max() <= 5.0)
        self.assertTrue(clipped["logp"].min() >= -0.5)
        self.assertTrue(clipped["molecular_weight"].max() <= 500.0)

    def test_graph_node_attributes(self):
        """feature 노드에 prior_low/prior_high/has_prior 속성이 설정됨."""
        df = _make_feature_df(50)
        self.ppb.apply(df)

        # tpsa 노드 확인
        tpsa_found = False
        for nid, ndata in self.G.nodes(data=True):
            if ndata.get("feature_key") == "tpsa":
                self.assertTrue(ndata.get("has_prior", False))
                self.assertEqual(ndata["prior_low"], 0)
                self.assertEqual(ndata["prior_high"], 90)
                tpsa_found = True
        self.assertTrue(tpsa_found, "tpsa feature node not found")

    def test_get_bounds_for_feature(self):
        """특정 feature의 bounds를 반환."""
        bounds = self.ppb.get_bounds_for_feature("logp")
        self.assertIsNotNone(bounds)
        self.assertEqual(bounds, (-0.5, 5.0))

        # 존재하지 않는 feature
        self.assertIsNone(self.ppb.get_bounds_for_feature("nonexistent"))

    def test_unmodified_columns(self):
        """Bounds가 없는 컬럼은 수정되지 않음."""
        df = _make_feature_df(50)
        original_hba = df["num_hba"].copy()
        clipped = self.ppb.apply(df)
        pd.testing.assert_series_equal(clipped["num_hba"], original_hba)


class TestStaticPheromone(unittest.TestCase):
    """StaticPheromone: ClinTox FDA EDT + Cramer 고정 페로몬."""

    def setUp(self):
        from src.aco.knowledge_heuristics import StaticPheromone
        self.G = _make_feature_graph()
        self.static_cfg = {
            "fda_edt_questions": [
                {"level": 0, "question": "New molecular entity?",
                 "feature": "molecular_weight", "threshold": 500},
                {"level": 1, "question": "Reactive groups?",
                 "feature": "num_halogens", "threshold": 3},
                {"level": 2, "question": "Pharmacophore toxicity?",
                 "feature": "num_aromatic_rings", "threshold": 4},
            ],
            "cramer_classes": {
                "class_I_low": {"molecular_weight_max": 200},
                "class_II_medium": {"molecular_weight_max": 400},
                "class_III_high": {"molecular_weight_min": 400},
            },
            "pheromone_multiplier": 5.0,
        }
        self.sp = StaticPheromone(
            self.static_cfg, self.G, pheromone_multiplier=5.0
        )

    def test_apply_creates_edt_nodes(self):
        """apply() 호출 후 EDT 노드가 그래프에 생성됨."""
        df = _make_feature_df(30)
        self.sp.apply(df)
        static_nodes = self.sp.get_static_nodes()
        self.assertGreater(len(static_nodes), 0)

        # EDT 노드 확인
        edt_count = sum(1 for n in static_nodes if n.startswith("edt:"))
        self.assertEqual(edt_count, 3)  # 3 EDT questions

    def test_apply_creates_cramer_nodes(self):
        """Cramer 3단계 노드가 생성됨."""
        df = _make_feature_df(30)
        self.sp.apply(df)
        static_nodes = self.sp.get_static_nodes()

        cramer_count = sum(1 for n in static_nodes if n.startswith("cramer:"))
        self.assertEqual(cramer_count, 3)

    def test_is_static_node(self):
        """고정 페로몬 노드 식별."""
        df = _make_feature_df(30)
        self.sp.apply(df)
        self.assertTrue(self.sp.is_static_node("edt:L0_molecular_weight"))
        self.assertTrue(self.sp.is_static_node("cramer:class_I_low"))
        self.assertFalse(self.sp.is_static_node("Thing"))

    def test_static_pheromone_values(self):
        """고정 페로몬 엣지의 pheromone 값이 배율에 따라 설정됨."""
        df = _make_feature_df(30)
        self.sp.apply(df)

        # EDT L0 노드의 연결 엣지들
        edt_node = "edt:L0_molecular_weight"
        if edt_node in self.G:
            edges = list(self.G.in_edges(edt_node, data=True))
            for u, v, edata in edges:
                if edata.get("static"):
                    self.assertGreater(edata["pheromone"], 1.0)

    def test_protect_from_evaporation(self):
        """protect_from_evaporation()으로 페로몬 복원."""
        df = _make_feature_df(30)
        self.sp.apply(df)

        # 인위적으로 페로몬 감소 시뮬레이션
        for nid in self.sp.get_static_nodes():
            if nid in self.G:
                for u, v, edata in self.G.edges(nid, data=True):
                    edata["pheromone"] = 0.1  # 감소

        self.sp.protect_from_evaporation()

        # 복원 확인
        for nid in self.sp.get_static_nodes():
            if nid in self.G:
                for u, v, edata in self.G.edges(nid, data=True):
                    self.assertGreaterEqual(edata["pheromone"], 5.0)


class TestBindingPocketHeuristic(unittest.TestCase):
    """BindingPocketHeuristic: BACE 바인딩 포켓 이진 특성."""

    def setUp(self):
        from src.aco.knowledge_heuristics import BindingPocketHeuristic
        self.G = _make_feature_graph()
        self.pocket_cfg = {
            "s1_occupancy": {
                "proxy_features": ["num_aromatic_rings", "logp"],
            },
            "s3_occupancy": {
                "proxy_features": ["num_hba", "tpsa"],
            },
            "asp32_interaction": {
                "proxy_features": ["num_hbd", "num_nitrogens"],
            },
            "asp228_interaction": {
                "proxy_features": ["num_oxygens", "num_hba"],
            },
        }
        self.bph = BindingPocketHeuristic(self.pocket_cfg, self.G)

    def test_apply_adds_binary_columns(self):
        """apply() 후 4개 binary 특성 컬럼이 추가됨."""
        df = _make_feature_df(50)
        result_df = self.bph.apply(df)

        expected_cols = [
            "pocket_s1_occupancy",
            "pocket_s3_occupancy",
            "pocket_asp32_interaction",
            "pocket_asp228_interaction",
        ]
        for col in expected_cols:
            self.assertIn(col, result_df.columns, f"Missing column: {col}")

    def test_binary_values(self):
        """이진 특성의 값이 0 또는 1."""
        df = _make_feature_df(50)
        result_df = self.bph.apply(df)

        for col in self.bph.get_binary_features():
            unique_vals = set(result_df[col].unique())
            self.assertTrue(
                unique_vals.issubset({0, 1}),
                f"{col} has non-binary values: {unique_vals}"
            )

    def test_pocket_nodes_in_graph(self):
        """포켓 feature 노드가 그래프에 생성됨."""
        df = _make_feature_df(50)
        self.bph.apply(df)

        for pocket_name in self.pocket_cfg:
            node_id = f"pocket:{pocket_name}"
            self.assertIn(node_id, self.G, f"Missing pocket node: {node_id}")
            ndata = self.G.nodes[node_id]
            self.assertTrue(ndata.get("is_feature", False))
            self.assertTrue(ndata.get("is_pocket_feature", False))

    def test_compute_pocket_eta(self):
        """포켓 노드의 eta가 3x 증폭됨."""
        df = _make_feature_df(50)
        self.bph.apply(df)

        # 포켓 노드
        eta_pocket = self.bph.compute_pocket_eta("pocket:s1_occupancy", 1.0)
        self.assertAlmostEqual(eta_pocket, 3.0)

        # 일반 노드
        eta_normal = self.bph.compute_pocket_eta("Thing", 1.0)
        self.assertAlmostEqual(eta_normal, 1.0)

    def test_get_binary_features(self):
        """생성된 이진 특성 리스트 반환."""
        df = _make_feature_df(50)
        self.bph.apply(df)
        features = self.bph.get_binary_features()
        self.assertEqual(len(features), 4)


class TestKnowledgeHeuristicsUnified(unittest.TestCase):
    """KnowledgeHeuristics: 통합 인터페이스 자동 전략 선택."""

    def test_physicochemical_profile(self):
        """physicochemical_prior 프로파일 → PriorBounds 활성화."""
        from src.aco.knowledge_heuristics import KnowledgeHeuristics
        G = _make_feature_graph()
        cfg = {
            "heuristic_profile": "physicochemical_prior",
            "prior_bounds": {
                "tpsa": {"low": 0, "high": 90},
            },
        }
        kh = KnowledgeHeuristics(cfg, G)
        self.assertTrue(kh.has_prior_bounds)
        self.assertFalse(kh.has_static_pheromone)
        self.assertFalse(kh.has_binding_pocket)
        self.assertEqual(kh.profile, "physicochemical_prior")

    def test_static_pheromone_profile(self):
        """fda_cramer_static 프로파일 → StaticPheromone 활성화."""
        from src.aco.knowledge_heuristics import KnowledgeHeuristics
        G = _make_feature_graph()
        cfg = {
            "heuristic_profile": "fda_cramer_static",
            "static_pheromone": {
                "fda_edt_questions": [],
                "cramer_classes": {},
                "pheromone_multiplier": 5.0,
            },
        }
        kh = KnowledgeHeuristics(cfg, G)
        self.assertTrue(kh.has_static_pheromone)
        self.assertFalse(kh.has_prior_bounds)

    def test_binding_pocket_profile(self):
        """binding_pocket 프로파일 → BindingPocket 활성화."""
        from src.aco.knowledge_heuristics import KnowledgeHeuristics
        G = _make_feature_graph()
        cfg = {
            "heuristic_profile": "binding_pocket",
            "pocket_features": {
                "s1": {"proxy_features": ["logp"]},
            },
        }
        kh = KnowledgeHeuristics(cfg, G)
        self.assertTrue(kh.has_binding_pocket)
        self.assertFalse(kh.has_static_pheromone)

    def test_default_profile(self):
        """default 프로파일 → 모든 전략 비활성."""
        from src.aco.knowledge_heuristics import KnowledgeHeuristics
        G = _make_feature_graph()
        cfg = {"heuristic_profile": "default"}
        kh = KnowledgeHeuristics(cfg, G)
        self.assertFalse(kh.has_prior_bounds)
        self.assertFalse(kh.has_static_pheromone)
        self.assertFalse(kh.has_binding_pocket)

    def test_apply_passthrough_default(self):
        """default 프로파일에서 apply()는 DataFrame을 그대로 반환."""
        from src.aco.knowledge_heuristics import KnowledgeHeuristics
        G = _make_feature_graph()
        cfg = {"heuristic_profile": "default"}
        kh = KnowledgeHeuristics(cfg, G)
        df = _make_feature_df(20)
        result = kh.apply(df)
        pd.testing.assert_frame_equal(result, df)

    def test_get_eta_modifier_prior(self):
        """prior_bounds 노드에 eta 1.5x 보너스."""
        from src.aco.knowledge_heuristics import KnowledgeHeuristics
        G = _make_feature_graph()
        cfg = {
            "heuristic_profile": "physicochemical_prior",
            "prior_bounds": {"tpsa": {"low": 0, "high": 90}},
        }
        kh = KnowledgeHeuristics(cfg, G)
        df = _make_feature_df(20)
        kh.apply(df)

        # tpsa feature 노드 찾기
        tpsa_node = None
        for nid, ndata in G.nodes(data=True):
            if ndata.get("feature_key") == "tpsa":
                tpsa_node = nid
                break
        self.assertIsNotNone(tpsa_node)
        eta = kh.get_eta_modifier(tpsa_node, base_eta=1.0)
        self.assertGreater(eta, 1.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. HierarchicalSearch 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHierarchicalSearch(unittest.TestCase):
    """HierarchicalSearch: 게이트 설치, 필터링, 엔트로피."""

    def _make_hierarchy_config(self):
        return {
            "source_ontology": "test",
            "gate_levels": [
                {"depth": 0, "concept": "root_concept"},
                {"depth": 1, "concept": "mid_concept"},
                {"depth": 2, "concept": "leaf_concept"},
            ],
            "max_depth_without_gate": 2,
            "gate_pheromone_bonus": 3.0,
        }

    def test_install_gates(self):
        """3개 게이트가 정상 설치됨."""
        from src.aco.hierarchical_search import HierarchicalSearch
        G = _make_mini_graph(15)
        cfg = self._make_hierarchy_config()
        hs = HierarchicalSearch(cfg, G)
        n_installed = hs.install_gates()

        self.assertEqual(n_installed, 3)
        self.assertEqual(len(hs.gates), 3)
        self.assertEqual(len(hs.gate_node_ids), 3)

    def test_gate_nodes_in_graph(self):
        """게이트 노드가 그래프에 is_gate 속성으로 존재."""
        from src.aco.hierarchical_search import HierarchicalSearch
        G = _make_mini_graph(15)
        cfg = self._make_hierarchy_config()
        hs = HierarchicalSearch(cfg, G)
        hs.install_gates()

        for gate_nid in hs.gate_node_ids:
            self.assertIn(gate_nid, G)
            self.assertTrue(G.nodes[gate_nid].get("is_gate", False))

    def test_gate_pheromone_bonus(self):
        """게이트 노드 엣지에 보너스 페로몬이 설정됨."""
        from src.aco.hierarchical_search import HierarchicalSearch
        G = _make_mini_graph(15)
        cfg = self._make_hierarchy_config()
        hs = HierarchicalSearch(cfg, G)
        hs.install_gates()

        for gate_nid in hs.gate_node_ids:
            for u, v, edata in G.edges(gate_nid, data=True):
                # _link_gates creates inter-gate edges at 0.8 * bonus
                self.assertGreaterEqual(edata.get("pheromone", 0), 2.0)

    def test_filter_candidates_no_gates_visited(self):
        """게이트 미통과 시 게이트 zone 내 노드가 필터링됨."""
        from src.aco.hierarchical_search import HierarchicalSearch
        G = _make_mini_graph(15)
        cfg = self._make_hierarchy_config()
        hs = HierarchicalSearch(cfg, G)
        hs.install_gates()

        visited_gates = set()
        candidates = list(G.nodes)
        filtered = hs.filter_candidates("Thing", candidates, visited_gates)
        # 게이트는 항상 접근 가능하므로 필터에 포함됨
        for gate_nid in hs.gate_node_ids:
            self.assertIn(gate_nid, filtered)

    def test_filter_candidates_with_gate_visited(self):
        """게이트 통과 후 더 많은 후보에 접근 가능."""
        from src.aco.hierarchical_search import HierarchicalSearch
        G = _make_mini_graph(15)
        cfg = self._make_hierarchy_config()
        hs = HierarchicalSearch(cfg, G)
        hs.install_gates()

        gate_0 = list(hs.gate_node_ids)[0]
        visited_no = set()
        visited_yes = {gate_0}

        candidates = list(G.nodes)
        f_no = hs.filter_candidates("Thing", candidates, visited_no)
        f_yes = hs.filter_candidates("Thing", candidates, visited_yes)

        # 게이트 통과 후 접근 가능 후보가 같거나 많음
        self.assertGreaterEqual(len(f_yes), len(f_no))

    def test_update_visited_gates(self):
        """게이트 노드 방문 시 visited_gates가 업데이트됨."""
        from src.aco.hierarchical_search import HierarchicalSearch
        G = _make_mini_graph(15)
        cfg = self._make_hierarchy_config()
        hs = HierarchicalSearch(cfg, G)
        hs.install_gates()

        visited = set()
        gate_nid = list(hs.gate_node_ids)[0]
        hs.update_visited_gates(gate_nid, visited)
        self.assertIn(gate_nid, visited)

        # 일반 노드 방문은 변화 없음
        hs.update_visited_gates("Thing", visited)
        self.assertNotIn("Thing", visited)

    def test_grant_gate_bonus(self):
        """경로에 게이트가 포함되면 보너스 > 1.0."""
        from src.aco.hierarchical_search import HierarchicalSearch
        G = _make_mini_graph(15)
        cfg = self._make_hierarchy_config()
        hs = HierarchicalSearch(cfg, G)
        hs.install_gates()

        gate_ids = list(hs.gate_node_ids)

        # 게이트 없는 경로
        bonus_none = hs.grant_gate_bonus(["Thing", "node_0", "node_1"])
        self.assertAlmostEqual(bonus_none, 1.0)

        # 게이트 1개 포함 경로
        bonus_one = hs.grant_gate_bonus(["Thing", gate_ids[0], "node_0"])
        self.assertGreater(bonus_one, 1.0)

        # 게이트 2개 포함 경로 → 더 높은 보너스
        bonus_two = hs.grant_gate_bonus(
            ["Thing", gate_ids[0], gate_ids[1], "node_0"]
        )
        self.assertGreater(bonus_two, bonus_one)

    def test_compute_entropy(self):
        """엔트로피가 0 이상, finite."""
        from src.aco.hierarchical_search import HierarchicalSearch
        G = _make_mini_graph(15)
        cfg = self._make_hierarchy_config()
        hs = HierarchicalSearch(cfg, G)
        hs.install_gates()

        entropy = hs.compute_entropy()
        self.assertGreaterEqual(entropy, 0.0)
        self.assertTrue(math.isfinite(entropy))

    def test_compute_entropy_empty(self):
        """게이트 없는 경우 엔트로피 = 0."""
        from src.aco.hierarchical_search import HierarchicalSearch
        G = _make_mini_graph(5)
        cfg = {
            "gate_levels": [],
            "max_depth_without_gate": 2,
            "gate_pheromone_bonus": 3.0,
        }
        hs = HierarchicalSearch(cfg, G)
        hs.install_gates()
        self.assertAlmostEqual(hs.compute_entropy(), 0.0)

    def test_gate_node_repr(self):
        """GateNode __repr__ 형식."""
        from src.aco.hierarchical_search import GateNode
        gate = GateNode("gate:test", 0, "test_concept", 2.0)
        r = repr(gate)
        self.assertIn("gate:test", r)
        self.assertIn("test_concept", r)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. DataPipeline 유틸리티 함수 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestStratifiedSplit(unittest.TestCase):
    """stratified_split: 계층화 분할."""

    def test_basic_split(self):
        """기본 분할: train/test 크기, 라벨 분포."""
        from src.aco.data_pipeline import stratified_split
        df = _make_feature_df(100)
        labels = _make_labels(100)

        train_df, test_df, train_labels, test_labels = stratified_split(
            df, labels, test_size=0.2, seed=42
        )

        # 크기 확인
        total = len(train_labels) + len(test_labels)
        self.assertEqual(total, 100)
        self.assertAlmostEqual(
            len(test_labels) / total, 0.2, delta=0.05
        )

        # label 타입 확인
        self.assertIsInstance(train_labels, np.ndarray)
        self.assertIsInstance(test_labels, np.ndarray)

    def test_stratified_class_ratio(self):
        """계층화: 클래스 비율이 train/test에 보존됨."""
        from src.aco.data_pipeline import stratified_split
        df = _make_feature_df(200)
        # 불균형 라벨 (30% positive)
        rng = np.random.RandomState(42)
        labels = (rng.random(200) < 0.3).astype(int)
        orig_ratio = labels.mean()

        _, _, train_l, test_l = stratified_split(
            df, labels, test_size=0.2
        )
        train_ratio = train_l.mean()
        test_ratio = test_l.mean()

        # 비율이 크게 벗어나지 않음 (±10%)
        self.assertAlmostEqual(train_ratio, orig_ratio, delta=0.10)
        self.assertAlmostEqual(test_ratio, orig_ratio, delta=0.15)

    def test_list_labels_input(self):
        """Python list 형태의 labels도 처리 가능."""
        from src.aco.data_pipeline import stratified_split
        df = _make_feature_df(50)
        labels_list = [0, 1] * 25  # list, not ndarray

        train_df, test_df, train_labels, test_labels = stratified_split(
            df, labels_list, test_size=0.2
        )
        self.assertEqual(len(train_labels) + len(test_labels), 50)

    def test_no_data_leakage(self):
        """train과 test 인덱스가 겹치지 않음."""
        from src.aco.data_pipeline import stratified_split
        df = _make_feature_df(100)
        labels = _make_labels(100)

        # DataFrame에 unique ID 추가
        df["uid"] = range(100)
        train_df, test_df, _, _ = stratified_split(df, labels, test_size=0.2)

        train_uids = set(train_df["uid"].values)
        test_uids = set(test_df["uid"].values)
        self.assertEqual(len(train_uids & test_uids), 0)


class TestComputeMetrics(unittest.TestCase):
    """compute_metrics: 평가 메트릭 계산."""

    def test_perfect_prediction(self):
        """완벽 예측 → accuracy=1.0, f1=1.0."""
        from src.aco.data_pipeline import compute_metrics
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0])
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["accuracy"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)
        self.assertAlmostEqual(m["precision"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)

    def test_all_wrong(self):
        """전부 틀린 예측 → accuracy=0.0."""
        from src.aco.data_pipeline import compute_metrics
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        m = compute_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["accuracy"], 0.0)

    def test_balanced_accuracy(self):
        """balanced_accuracy 계산."""
        from src.aco.data_pipeline import compute_metrics
        # 불균형: 8 negatives, 2 positives
        y_true = np.array([0]*8 + [1]*2)
        y_pred = np.array([0]*10)  # 전부 0 예측
        m = compute_metrics(y_true, y_pred)
        # accuracy = 0.8, balanced_accuracy = (1.0 + 0.0) / 2 = 0.5
        self.assertAlmostEqual(m["accuracy"], 0.8)
        self.assertAlmostEqual(m["balanced_accuracy"], 0.5)

    def test_auc_roc_with_proba(self):
        """AUC-ROC 계산 (proba 제공 시)."""
        from src.aco.data_pipeline import compute_metrics
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        m = compute_metrics(y_true, y_pred, y_proba)
        self.assertIn("auc_roc", m)
        self.assertGreater(m["auc_roc"], 0.5)

    def test_empty_predictions(self):
        """빈 배열 → accuracy=0.0."""
        from src.aco.data_pipeline import compute_metrics
        m = compute_metrics(np.array([]), np.array([]))
        self.assertAlmostEqual(m["accuracy"], 0.0)


class TestLoadDataset(unittest.TestCase):
    """load_dataset: CSV 로드 및 전처리."""

    def test_load_bbbp(self):
        """BBBP 로드 → SMILES 리스트 + 라벨."""
        from src.aco.data_pipeline import load_dataset
        base_dir = Path(ROOT)
        csv_path = base_dir / "data" / "bbbp" / "BBBP.csv"
        if not csv_path.exists():
            self.skipTest("BBBP CSV not found")

        cfg = {
            "file": "data/bbbp/BBBP.csv",
            "smiles_col": "smiles",
            "label_col": "p_np",
        }
        smiles, labels, target = load_dataset("bbbp", cfg, base_dir)
        self.assertGreater(len(smiles), 100)
        self.assertEqual(len(smiles), len(labels))
        self.assertEqual(target, "p_np")
        self.assertIsInstance(labels, np.ndarray)
        # 모든 라벨이 0 또는 1
        self.assertTrue(set(np.unique(labels)).issubset({0, 1}))

    def test_load_with_max_samples(self):
        """max_samples 제한."""
        from src.aco.data_pipeline import load_dataset
        base_dir = Path(ROOT)
        csv_path = base_dir / "data" / "bbbp" / "BBBP.csv"
        if not csv_path.exists():
            self.skipTest("BBBP CSV not found")

        cfg = {
            "file": "data/bbbp/BBBP.csv",
            "smiles_col": "smiles",
            "label_col": "p_np",
        }
        smiles, labels, _ = load_dataset(
            "bbbp", cfg, base_dir, max_samples=100
        )
        self.assertLessEqual(len(smiles), 100)

    def test_load_invalid_target(self):
        """존재하지 않는 타겟 → ValueError."""
        from src.aco.data_pipeline import load_dataset
        base_dir = Path(ROOT)
        csv_path = base_dir / "data" / "bbbp" / "BBBP.csv"
        if not csv_path.exists():
            self.skipTest("BBBP CSV not found")

        cfg = {
            "file": "data/bbbp/BBBP.csv",
            "smiles_col": "smiles",
            "label_col": "p_np",
        }
        with self.assertRaises(ValueError):
            load_dataset("bbbp", cfg, base_dir, target="nonexistent_col")

    def test_load_tox21_multi_target(self):
        """Tox21 멀티타겟 NR-AhR 로드."""
        from src.aco.data_pipeline import load_dataset
        base_dir = Path(ROOT)
        csv_path = base_dir / "data" / "tox21" / "tox21.csv"
        if not csv_path.exists():
            self.skipTest("Tox21 CSV not found")

        cfg = {
            "file": "data/tox21/tox21.csv",
            "smiles_col": "smiles",
            "label_col": "NR-AhR",
        }
        smiles, labels, target = load_dataset(
            "tox21", cfg, base_dir, target="NR-AhR"
        )
        self.assertEqual(target, "NR-AhR")
        self.assertGreater(len(smiles), 100)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 통합 테스트: validate_graph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPipelineValidateGraph(unittest.TestCase):
    """ToxicityDataPipeline.validate_graph — 학습 생략, 그래프 검증."""

    @classmethod
    def setUpClass(cls):
        config_path = Path(ROOT) / "configs" / "dataset_ontology_config.json"
        dto_path = Path(ROOT) / "ontology" / "DTO.xrdf"
        bbbp_path = Path(ROOT) / "data" / "bbbp" / "BBBP.csv"
        cls.can_run = (
            config_path.exists()
            and dto_path.exists()
            and bbbp_path.exists()
        )
        if cls.can_run:
            from src.aco.data_pipeline import ToxicityDataPipeline
            from src.aco.ontology_router import _clear_cache
            _clear_cache()
            cls.pipeline = ToxicityDataPipeline(
                config_path=str(config_path),
                base_dir=str(Path(ROOT)),
            )

    def test_validate_bbbp(self):
        """BBBP validate_graph: 그래프 생성 성공."""
        if not self.can_run:
            self.skipTest("Required files not found")
        result = self.pipeline.validate_graph("bbbp")
        self.assertEqual(result["status"], "graph_only")
        self.assertEqual(result["dataset"], "bbbp")
        self.assertGreater(result["n_samples"], 100)
        self.assertGreater(result["n_graph_nodes"], 50)
        self.assertGreater(result["n_graph_edges"], 50)
        self.assertGreater(result["n_feature_nodes"], 0)
        self.assertEqual(result["heuristic_profile"], "physicochemical_prior")

    def test_validate_bace(self):
        """BACE validate_graph: binding_pocket 프로파일 확인."""
        bace_path = Path(ROOT) / "data" / "bace" / "bace.csv"
        if not self.can_run or not bace_path.exists():
            self.skipTest("Required files not found")
        result = self.pipeline.validate_graph("bace")
        self.assertEqual(result["heuristic_profile"], "binding_pocket")
        self.assertGreater(result["n_features"], 10)  # SMILES desc + extra cols

    def test_validate_clintox(self):
        """ClinTox validate_graph: fda_cramer_static 프로파일."""
        clintox_path = Path(ROOT) / "data" / "clintox" / "clintox.csv"
        if not self.can_run or not clintox_path.exists():
            self.skipTest("Required files not found")
        result = self.pipeline.validate_graph("clintox", target="CT_TOX")
        self.assertEqual(result["heuristic_profile"], "fda_cramer_static")
        self.assertEqual(result["target"], "CT_TOX")

    def test_validate_returns_ontologies(self):
        """결과에 사용된 온톨로지 목록이 포함됨."""
        if not self.can_run:
            self.skipTest("Required files not found")
        result = self.pipeline.validate_graph("bbbp")
        self.assertIn("ontologies", result)
        self.assertIn("chebi", result["ontologies"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 70)
    print(" ACO-Semantic-Forest: Data Pipeline Unit Tests")
    print("=" * 70)
    unittest.main(verbosity=2)
