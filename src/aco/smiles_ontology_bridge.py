"""
Step 2: SMILES descriptor <-> ontology node bridge mapper

Maps SMILES DataFrame feature column names to ontology graph nodes.
Matched nodes get ``is_feature=True`` so ants can compute split conditions.

**Bridging Strategies (domain-specific)**

1) ChEBI-based ontology (BBBP, ClinTox):
   Real physicochemical property nodes exist in the ontology.
   Use synonym dict + fuzzy matching for direct is_feature=True mapping.

2) GO / MeSH / DTO / BAO (Tox21, SIDER, BACE, HIV):
   Target/disease/bioassay centric -- no SMILES feature nodes.
   Create a single *anchor* node 'Molecular Structural Features'
   under root and flat-connect all 16 features to it.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

# -- fuzzy ratio ----------------------------------------------------------

try:
    from fuzzywuzzy import fuzz as _fuzz

    def _fuzzy_ratio(a: str, b: str) -> int:
        return _fuzz.token_set_ratio(a.lower(), b.lower())

except ImportError:
    _fuzz = None  # type: ignore[assignment]

    def _fuzzy_ratio(a: str, b: str) -> int:  # type: ignore[misc]
        a_low, b_low = a.lower(), b.lower()
        if a_low == b_low:
            return 100
        if a_low in b_low or b_low in a_low:
            return 80
        ta = set(re.split(r"[\\W_]+", a_low))
        tb = set(re.split(r"[\\W_]+", b_low))
        ta.discard(""); tb.discard("")
        if not ta or not tb:
            return 0
        overlap = len(ta & tb) / max(len(ta | tb), 1)
        return int(overlap * 100)


# -- known numeric features -----------------------------------------------

KNOWN_NUMERIC_FEATURES: Dict[str, List[str]] = {
    "molecular_weight": ["MolWt", "MW", "molecular_weight", "MolecularWeight"],
    "logp": ["LogP", "logP", "MolLogP", "CLogP", "logp"],
    "tpsa": ["TPSA", "tpsa", "TopologicalPolarSurfaceArea"],
    "num_hbd": ["NumHDonors", "HBD", "num_hbd", "HydrogenBondDonors"],
    "num_hba": ["NumHAcceptors", "HBA", "num_hba", "HydrogenBondAcceptors"],
    "num_rotatable_bonds": ["NumRotatableBonds", "RotBonds", "num_rotatable_bonds"],
    "num_rings": ["RingCount", "NumRings", "num_rings"],
    "num_aromatic_rings": ["NumAromaticRings", "AromaticRings", "num_aromatic_rings"],
    "num_atoms": ["NumAtoms", "num_atoms", "AtomCount"],
    "num_heavy_atoms": ["HeavyAtomCount", "num_heavy_atoms"],
    "molar_refractivity": ["MolarRefractivity", "MolMR", "molar_refractivity"],
    "n_plus_o_count": ["NOCount", "n_plus_o_count"],
    "num_heteroatoms": ["NumHeteroatoms", "num_heteroatoms"],
    "formal_charge": ["FormalCharge", "formal_charge"],
    "fsp3": ["FractionCSP3", "fsp3", "Fsp3"],
    "num_carbons": ["NumCarbons", "num_carbons", "CarbonCount"],
    "num_oxygens": ["NumOxygens", "num_oxygens", "OxygenCount"],
    "num_nitrogens": ["NumNitrogens", "num_nitrogens", "NitrogenCount"],
    "num_halogens": ["NumHalogens", "num_halogens", "HalogenCount"],
}

_SYNONYM_TO_KEY: Dict[str, str] = {}
for _key, _syns in KNOWN_NUMERIC_FEATURES.items():
    for _syn in _syns:
        _SYNONYM_TO_KEY[_syn.lower()] = _key
    _SYNONYM_TO_KEY[_key.lower()] = _key


def _normalize_feature_name(name: str) -> Optional[str]:
    return _SYNONYM_TO_KEY.get(name.lower())


# -- ChEBI synonym dict ---------------------------------------------------

_CHEBI_SYNONYM_DICT: Dict[str, List[str]] = {
    "molecular_weight": [
        "molecular mass", "mass", "molecular entity",
        "has_molecular_mass", "monoisotopic mass",
    ],
    "logp": [
        "partition coefficient", "lipophilicity",
        "octanol-water partition coefficient", "hydrophobicity",
        "organic molecular entity",
    ],
    "tpsa": [
        "polar surface area", "topological polar surface area",
        "polar", "polarity",
    ],
    "num_hbd": [
        "hydrogen bond donor", "hydrogen-bond donor",
        "h-bond donor", "proton donor",
    ],
    "num_hba": [
        "hydrogen bond acceptor", "hydrogen-bond acceptor",
        "h-bond acceptor", "proton acceptor",
    ],
    "num_rotatable_bonds": [
        "rotatable bond", "single bond", "conformational flexibility",
    ],
    "num_rings": [
        "ring", "cyclic compound", "cyclic", "ring system",
        "carbocyclic compound",
    ],
    "num_aromatic_rings": [
        "aromatic compound", "aromatic ring", "arene",
        "aromatic", "benzenoid",
    ],
    "num_atoms": ["atom", "heavy atom", "atom count"],
    "num_heavy_atoms": ["heavy atom", "non-hydrogen atom"],
    "fsp3": [
        "sp3", "tetrahedral carbon", "saturated carbon",
        "fraction", "saturation",
    ],
    "num_heteroatoms": [
        "heteroatom", "heteroatomic", "heterocyclic compound",
    ],
    "num_carbons": ["carbon atom", "carbon", "organic compound"],
    "num_oxygens": ["oxygen atom", "oxygen", "oxide"],
    "num_nitrogens": [
        "nitrogen atom", "nitrogen", "amine",
        "organonitrogen compound",
    ],
    "num_halogens": [
        "halogen", "halide", "organohalogen compound",
        "fluorine", "chlorine", "bromine",
    ],
}

# -- anchor constants -----------------------------------------------------

ANCHOR_NODE_ID = "anchor:molecular_structural_features"
ANCHOR_LABEL = "Molecular Structural Features"
ANCHOR_URI = "urn:aco:molecular_structural_features"


class SMILESOntologyBridge:
    """SMILES feature <-> ontology node bridge.

    Strategies:
      chebi  -- synonym dict + fuzzy match to real ontology nodes
      anchor -- single anchor node under root, flat feature attachment

    Parameters
    ----------
    graph : nx.DiGraph
    match_threshold : int  (0-100, default 70)
    domain : str  ("chebi" or "anchor", default "anchor")
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        match_threshold: int = 70,
        domain: str = "anchor",
    ) -> None:
        self.graph = graph
        self.match_threshold = match_threshold
        self.domain = domain.lower()
        self.mapped_features: Dict[str, str] = {}
        self._label_to_node: Dict[str, str] = {}
        for nid, ndata in graph.nodes(data=True):
            label = ndata.get("label", nid)
            self._label_to_node[label.lower()] = nid

    # == public API ========================================================

    def auto_map(
        self,
        feature_names: List[str],
        *,
        extra_synonyms: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, str]:
        """Map feature names to ontology nodes (domain-aware)."""
        if self.domain == "chebi":
            return self._map_chebi(feature_names, extra_synonyms=extra_synonyms)
        else:
            return self._map_anchor(feature_names)

    def map_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        exclude_columns: Optional[Set[str]] = None,
    ) -> Dict[str, str]:
        exclude = exclude_columns or set()
        numeric_cols = [
            c for c in df.select_dtypes(include="number").columns
            if c not in exclude
        ]
        return self.auto_map(numeric_cols)

    def create_feature_nodes(
        self,
        feature_names: List[str],
        *,
        connect_to: Optional[str] = None,
    ) -> List[str]:
        """Create feature nodes for unmapped features (backward compat)."""
        created: List[str] = []
        G = self.graph
        parent = connect_to
        if parent is None:
            if ANCHOR_NODE_ID in G:
                parent = ANCHOR_NODE_ID
            else:
                parent = self._find_root()
        for feat in feature_names:
            if feat in self.mapped_features:
                nid = self.mapped_features[feat]
                G.nodes[nid]["is_feature"] = True
                G.nodes[nid]["feature_key"] = feat
                created.append(nid)
                continue
            norm = _normalize_feature_name(feat)
            key = norm if norm else feat.lower()
            nid = f"feature:{key}"
            self._create_flat_feature(
                nid, key, feat,
                parent if parent else ANCHOR_NODE_ID,
            )
            created.append(nid)
        return created

    def get_split_condition(
        self,
        node_id: str,
        feature_series: pd.Series,
        *,
        strategy: str = "median",
    ) -> Optional[Dict[str, Any]]:
        """Compute split condition at an is_feature node."""
        ndata = self.graph.nodes.get(node_id)
        if ndata is None or not ndata.get("is_feature"):
            return None
        feat_key = ndata.get("feature_key", node_id)
        values = feature_series.dropna()
        if values.empty:
            return None
        if strategy == "median":
            threshold = float(values.median())
        elif strategy == "mean":
            threshold = float(values.mean())
        elif strategy == "quartile":
            threshold = float(values.quantile(0.5))
        else:
            threshold = float(values.median())
        return {"feature": feat_key, "operator": "<=", "threshold": threshold}

    def print_mapping(self) -> None:
        print(
            f"[SMILESOntologyBridge] domain={self.domain}, "
            f"mapped={len(self.mapped_features)}"
        )
        for feat, nid in sorted(self.mapped_features.items()):
            is_feat = self.graph.nodes[nid].get("is_feature", False)
            print(f"  {feat:30s} -> {nid} (is_feature={is_feat})")

    def get_mapping_stats(self) -> Dict[str, Any]:
        G = self.graph
        feats = {nid for nid in G.nodes() if G.nodes[nid].get("is_feature")}
        return {
            "domain": self.domain,
            "feature_nodes": len(feats),
            "total_mapped": len(self.mapped_features),
            "has_anchor": ANCHOR_NODE_ID in G,
        }

    # == ChEBI strategy ====================================================

    def _map_chebi(
        self,
        feature_names: List[str],
        *,
        extra_synonyms: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, str]:
        """Map features to real ChEBI ontology nodes via synonym dict."""
        node_labels = list(self._label_to_node.keys())
        unmapped_for_anchor: List[str] = []

        for feat in feature_names:
            if feat in self.mapped_features:
                continue
            feat_lower = feat.lower()
            norm = _normalize_feature_name(feat)
            key = norm if norm else feat_lower
            matched = False

            # 1) ChEBI synonym dict
            synonyms = _CHEBI_SYNONYM_DICT.get(key, [])
            for syn in synonyms:
                if syn.lower() in self._label_to_node:
                    self._register(feat, self._label_to_node[syn.lower()])
                    matched = True
                    break
            if matched:
                continue

            # 2) exact match
            if feat_lower in self._label_to_node:
                self._register(feat, self._label_to_node[feat_lower])
                continue
            if norm and norm.lower() in self._label_to_node:
                self._register(feat, self._label_to_node[norm.lower()])
                continue

            # 3) extra_synonyms
            if extra_synonyms and feat in extra_synonyms:
                for syn in extra_synonyms[feat]:
                    if syn.lower() in self._label_to_node:
                        self._register(feat, self._label_to_node[syn.lower()])
                        matched = True
                        break
                if matched:
                    continue

            # 4) substring via synonyms
            for syn in synonyms:
                syn_lower = syn.lower()
                for nl in node_labels:
                    if syn_lower in nl or nl in syn_lower:
                        self._register(feat, self._label_to_node[nl])
                        matched = True
                        break
                if matched:
                    break
            if matched:
                continue

            # 5) fuzzy via synonyms
            best_score, best_node = 0, None
            for syn in synonyms:
                for nl in node_labels:
                    score = _fuzzy_ratio(syn, nl)
                    if score > best_score:
                        best_score, best_node = score, nl
            if best_score >= self.match_threshold and best_node is not None:
                self._register(feat, self._label_to_node[best_node])
                logger.info(
                    "ChEBI fuzzy '%s' -> '%s' (score=%d)",
                    feat, best_node, best_score,
                )
                continue

            # 6) anchor fallback
            logger.debug("ChEBI: no match for '%s' -> anchor fallback", feat)
            unmapped_for_anchor.append(feat)

        if unmapped_for_anchor:
            self._ensure_anchor()
            for feat in unmapped_for_anchor:
                norm = _normalize_feature_name(feat)
                key = norm if norm else feat.lower()
                nid = f"feature:{key}"
                self._create_flat_feature(nid, key, feat, ANCHOR_NODE_ID)

        return dict(self.mapped_features)

    # == Anchor strategy ===================================================

    def _map_anchor(self, feature_names: List[str]) -> Dict[str, str]:
        """Flat-connect all features under a single anchor node."""
        self._ensure_anchor()
        for feat in feature_names:
            if feat in self.mapped_features:
                continue
            norm = _normalize_feature_name(feat)
            key = norm if norm else feat.lower()
            nid = f"feature:{key}"
            self._create_flat_feature(nid, key, feat, ANCHOR_NODE_ID)
        return dict(self.mapped_features)

    # == internals =========================================================

    def _ensure_anchor(self) -> None:
        G = self.graph
        if ANCHOR_NODE_ID in G:
            return
        root = self._find_root()
        G.add_node(
            ANCHOR_NODE_ID, label=ANCHOR_LABEL,
            uri=ANCHOR_URI, is_anchor=True,
        )
        if root and root in G:
            G.add_edge(
                ANCHOR_NODE_ID, root,
                predicate="subClassOf", pheromone=1.0,
            )
            G.add_edge(
                root, ANCHOR_NODE_ID,
                predicate="hasSubClass", pheromone=1.0,
            )
        logger.info(
            "Anchor node '%s' created under root '%s'",
            ANCHOR_NODE_ID, root,
        )

    def _create_flat_feature(
        self, node_id, feature_key, original_name, parent_id,
    ) -> None:
        G = self.graph
        if node_id not in G:
            G.add_node(
                node_id, label=original_name,
                uri=f"urn:smiles_feature:{feature_key}",
                is_feature=True, feature_key=feature_key,
                feature_name=original_name,
            )
        else:
            G.nodes[node_id]["is_feature"] = True
            G.nodes[node_id]["feature_key"] = feature_key
            G.nodes[node_id]["feature_name"] = original_name
        if parent_id in G:
            if not G.has_edge(node_id, parent_id):
                G.add_edge(
                    node_id, parent_id,
                    predicate="subClassOf", pheromone=1.0,
                )
            if not G.has_edge(parent_id, node_id):
                G.add_edge(
                    parent_id, node_id,
                    predicate="hasFeature", pheromone=1.0,
                )
        self.mapped_features[original_name] = node_id
        self._label_to_node[feature_key.lower()] = node_id
        logger.debug(
            "Flat-mapped '%s' -> %s (parent=%s)",
            original_name, node_id, parent_id,
        )

    def _find_root(self) -> Optional[str]:
        G = self.graph
        for c in ("Thing", "thing", "owl:Thing"):
            if c in G:
                return c
        if G.number_of_nodes() > 0:
            return max(G.nodes, key=lambda n: G.degree(n))
        return None

    def _register(self, feature_name, node_id) -> None:
        self.mapped_features[feature_name] = node_id
        self.graph.nodes[node_id]["is_feature"] = True
        self.graph.nodes[node_id]["feature_key"] = feature_name
        logger.debug("Mapped '%s' -> node '%s'", feature_name, node_id)


# -- CLI demo -------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.aco.owl_graph_builder import OWLGraphBuilder
    logging.basicConfig(level=logging.INFO)

    owl_file = sys.argv[1] if len(sys.argv) > 1 else "ontology/DTO.xrdf"
    domain = sys.argv[2] if len(sys.argv) > 2 else "anchor"

    builder = OWLGraphBuilder(owl_file)
    G = builder.build()
    builder.print_summary()

    bridge = SMILESOntologyBridge(G, match_threshold=65, domain=domain)
    sample = [
        "molecular_weight", "logp", "tpsa", "num_hbd", "num_hba",
        "num_rotatable_bonds", "num_rings", "num_aromatic_rings",
        "num_atoms", "num_heavy_atoms", "fsp3", "num_heteroatoms",
        "num_carbons", "num_oxygens", "num_nitrogens", "num_halogens",
    ]
    mapping = bridge.auto_map(sample)
    unmapped = [f for f in sample if f not in mapping]
    if unmapped:
        bridge.create_feature_nodes(unmapped)
    bridge.print_mapping()
    print()
    print(f"Stats: {bridge.get_mapping_stats()}")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
