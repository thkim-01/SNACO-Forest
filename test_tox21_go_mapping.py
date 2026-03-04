"""
Task 3: Tox21 -> GO 매핑 검증 단위 테스트

Verifies:
  1. OntologyRouter routes Tox21 to GO (1:1)
  2. GO ontology loads (real or placeholder)
  3. Anchor node created under GO root
  4. All 16 SMILES features bridged under anchor
  5. "Mapped 16 features" — no fake hierarchy nodes
  6. Bridge domain is "anchor"
"""
import os
import sys
import unittest
from pathlib import Path

import networkx as nx

# Resolve project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


class TestTox21GORouting(unittest.TestCase):
    """OntologyRouter: Tox21 -> GO (1:1 strict routing)."""

    @classmethod
    def setUpClass(cls):
        from src.aco.ontology_router import OntologyRouter, _clear_cache
        _clear_cache()
        cls.config_path = Path(ROOT) / "configs" / "dataset_ontology_config.json"
        cls.base_dir = Path(ROOT)
        cls.router = OntologyRouter(
            config_path=str(cls.config_path),
            base_dir=str(cls.base_dir),
            use_cache=False,
        )

    def test_tox21_routes_to_go(self):
        """Tox21 maps to exactly ['go']."""
        required = self.router.get_required_ontologies("tox21")
        self.assertEqual(required, ["go"])

    def test_tox21_bridge_domain_is_anchor(self):
        """Tox21 bridge domain is 'anchor'."""
        self.assertEqual(self.router.get_bridge_domain("tox21"), "anchor")

    def test_tox21_not_dto(self):
        """Tox21 should NOT use DTO."""
        required = self.router.get_required_ontologies("tox21")
        self.assertNotIn("dto", required)

    def test_go_ontology_loads(self):
        """GO ontology loads as a non-empty graph."""
        G = self.router.route("tox21")
        self.assertIsInstance(G, nx.DiGraph)
        self.assertGreater(G.number_of_nodes(), 0)
        self.assertGreater(G.number_of_edges(), 0)


class TestTox21GOBridge(unittest.TestCase):
    """SMILESOntologyBridge: anchor strategy on GO graph for Tox21."""

    FEATURES_16 = [
        "molecular_weight", "logp", "tpsa", "num_hbd", "num_hba",
        "num_rotatable_bonds", "num_rings", "num_aromatic_rings",
        "num_atoms", "num_heavy_atoms", "fsp3", "num_heteroatoms",
        "num_carbons", "num_oxygens", "num_nitrogens", "num_halogens",
    ]

    @classmethod
    def setUpClass(cls):
        from src.aco.ontology_router import OntologyRouter, _clear_cache
        _clear_cache()
        config_path = Path(ROOT) / "configs" / "dataset_ontology_config.json"
        router = OntologyRouter(
            config_path=str(config_path),
            base_dir=str(ROOT),
            use_cache=False,
        )
        cls.G = router.route("tox21")
        cls.domain = router.get_bridge_domain("tox21")

    def test_anchor_domain(self):
        self.assertEqual(self.domain, "anchor")

    def test_bridge_maps_all_16_features(self):
        """Bridge maps all 16 features under anchor node."""
        from src.aco.smiles_ontology_bridge import SMILESOntologyBridge, ANCHOR_NODE_ID
        import copy

        G = copy.deepcopy(self.G)
        bridge = SMILESOntologyBridge(G, match_threshold=65, domain=self.domain)
        mapping = bridge.auto_map(self.FEATURES_16)
        unmapped = [f for f in self.FEATURES_16 if f not in mapping]
        if unmapped:
            bridge.create_feature_nodes(unmapped)

        # All 16 should be mapped
        self.assertEqual(len(bridge.mapped_features), 16,
                         f"Expected 16 mapped, got {len(bridge.mapped_features)}")

        # All 16 should be is_feature=True
        feature_nodes = {
            nid for nid in G.nodes()
            if G.nodes[nid].get("is_feature")
        }
        self.assertGreaterEqual(len(feature_nodes), 16)

    def test_anchor_node_exists(self):
        """Anchor node is created in the GO graph."""
        from src.aco.smiles_ontology_bridge import SMILESOntologyBridge, ANCHOR_NODE_ID
        import copy

        G = copy.deepcopy(self.G)
        bridge = SMILESOntologyBridge(G, match_threshold=65, domain="anchor")
        bridge.auto_map(self.FEATURES_16)

        self.assertIn(ANCHOR_NODE_ID, G)
        self.assertTrue(G.nodes[ANCHOR_NODE_ID].get("is_anchor"))

    def test_anchor_connected_to_root(self):
        """Anchor node has edge to/from some root node."""
        from src.aco.smiles_ontology_bridge import SMILESOntologyBridge, ANCHOR_NODE_ID
        import copy

        G = copy.deepcopy(self.G)
        bridge = SMILESOntologyBridge(G, match_threshold=65, domain="anchor")
        bridge.auto_map(self.FEATURES_16)

        # anchor should have at least one edge
        self.assertGreater(G.degree(ANCHOR_NODE_ID), 0)

    def test_features_flat_under_anchor(self):
        """All 16 feature nodes connect directly to anchor (flat, not hierarchy)."""
        from src.aco.smiles_ontology_bridge import SMILESOntologyBridge, ANCHOR_NODE_ID
        import copy

        G = copy.deepcopy(self.G)
        bridge = SMILESOntologyBridge(G, match_threshold=65, domain="anchor")
        bridge.auto_map(self.FEATURES_16)

        # Each feature node should have edge to/from anchor
        for feat, nid in bridge.mapped_features.items():
            has_edge = G.has_edge(nid, ANCHOR_NODE_ID) or G.has_edge(ANCHOR_NODE_ID, nid)
            self.assertTrue(has_edge,
                            f"Feature '{feat}' ({nid}) not connected to anchor")

    def test_no_fake_hierarchy_nodes(self):
        """No _SEMANTIC_CATEGORIES nodes (physicochemical_property etc.)."""
        from src.aco.smiles_ontology_bridge import SMILESOntologyBridge
        import copy

        G = copy.deepcopy(self.G)
        bridge = SMILESOntologyBridge(G, match_threshold=65, domain="anchor")
        bridge.auto_map(self.FEATURES_16)

        fake_labels = {
            "physicochemical_property", "hydrogen_bonding",
            "structural_topology", "atomic_composition",
            "molecular_descriptor",
        }
        for nid, ndata in G.nodes(data=True):
            label = ndata.get("label", "")
            self.assertNotIn(label.lower(), fake_labels,
                             f"Fake hierarchy node found: {nid} ({label})")

    def test_mapping_stats(self):
        """get_mapping_stats() returns correct info."""
        from src.aco.smiles_ontology_bridge import SMILESOntologyBridge, ANCHOR_NODE_ID
        import copy

        G = copy.deepcopy(self.G)
        bridge = SMILESOntologyBridge(G, match_threshold=65, domain="anchor")
        bridge.auto_map(self.FEATURES_16)

        stats = bridge.get_mapping_stats()
        self.assertEqual(stats["domain"], "anchor")
        self.assertTrue(stats["has_anchor"])
        self.assertEqual(stats["total_mapped"], 16)
        self.assertGreaterEqual(stats["feature_nodes"], 16)

    def test_print_mapping_output(self):
        """print_mapping() runs without error and shows 'Mapped 16'."""
        from src.aco.smiles_ontology_bridge import SMILESOntologyBridge
        import copy
        import io
        from contextlib import redirect_stdout

        G = copy.deepcopy(self.G)
        bridge = SMILESOntologyBridge(G, match_threshold=65, domain="anchor")
        bridge.auto_map(self.FEATURES_16)

        buf = io.StringIO()
        with redirect_stdout(buf):
            bridge.print_mapping()

        output = buf.getvalue()
        self.assertIn("mapped=16", output)
        self.assertIn("anchor", output)


class TestAllDatasetsRouting(unittest.TestCase):
    """Verify 1:1 routing for all 6 datasets."""

    EXPECTED = {
        "bbbp":    ("chebi",  "chebi"),
        "hiv":     ("bao",    "anchor"),
        "clintox": ("chebi",  "chebi"),
        "bace":    ("dto",    "anchor"),
        "sider":   ("mesh",   "anchor"),
        "tox21":   ("go",     "anchor"),
    }

    @classmethod
    def setUpClass(cls):
        from src.aco.ontology_router import OntologyRouter, _clear_cache
        _clear_cache()
        config_path = Path(ROOT) / "configs" / "dataset_ontology_config.json"
        cls.router = OntologyRouter(
            config_path=str(config_path),
            base_dir=str(ROOT),
            use_cache=False,
        )

    def test_all_datasets_present(self):
        self.assertEqual(
            set(self.router.available_datasets),
            set(self.EXPECTED.keys()),
        )

    def test_routing_is_1_to_1(self):
        """Each dataset routes to exactly 1 ontology."""
        for ds, (onto, domain) in self.EXPECTED.items():
            required = self.router.get_required_ontologies(ds)
            self.assertEqual(len(required), 1,
                             f"{ds}: expected 1 ontology, got {required}")
            self.assertEqual(required[0], onto,
                             f"{ds}: expected '{onto}', got '{required[0]}'")

    def test_bridge_domains(self):
        """Bridge domain matches expected."""
        for ds, (onto, domain) in self.EXPECTED.items():
            actual = self.router.get_bridge_domain(ds)
            self.assertEqual(actual, domain,
                             f"{ds}: expected domain '{domain}', got '{actual}'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
