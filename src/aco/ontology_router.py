"""
OntologyRouter -- strict 1:1 dataset -> ontology routing

Each dataset is routed to exactly ONE domain-specific ontology:
  BBBP     -> ChEBI    (bridge: chebi)
  HIV      -> BAO      (bridge: anchor)
  ClinTox  -> ChEBI    (bridge: chebi)
  BACE     -> DTO      (bridge: anchor)
  SIDER    -> MeSH     (bridge: anchor)
  Tox21    -> GO       (bridge: anchor)

Large ontologies (ChEBI 773 MB, GO 123 MB, MeSH 185 MB, BAO) are
parsed with a triple-count ceiling to control memory.

Usage:
    router = OntologyRouter(config_path="configs/dataset_ontology_config.json")
    G = router.route("tox21")           # loads GO only
    domain = router.get_bridge_domain("tox21")  # "anchor"
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from .owl_graph_builder import OWLGraphBuilder

logger = logging.getLogger(__name__)

# -- global cache ----------------------------------------------------------
_ONTOLOGY_CACHE: Dict[str, nx.DiGraph] = {}


def _clear_cache() -> None:
    """Clear the global ontology cache (for testing)."""
    _ONTOLOGY_CACHE.clear()


class OntologyRouter:
    """Strict 1:1 dataset -> ontology router.

    Parameters
    ----------
    config_path : str | Path
        Path to ``configs/dataset_ontology_config.json``.
    base_dir : str | Path | None
        Project root. Defaults to config file grandparent.
    use_cache : bool
        Cache parsed ontologies to avoid re-parsing. Default True.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/dataset_ontology_config.json",
        base_dir: Optional[str | Path] = None,
        use_cache: bool = True,
    ) -> None:
        self.config_path = Path(config_path)
        if base_dir is None:
            self.base_dir = self.config_path.resolve().parent.parent
        else:
            self.base_dir = Path(base_dir).resolve()
        self.use_cache = use_cache

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config: Dict[str, Any] = json.load(f)

        self._datasets = self._config.get("datasets", {})
        self._ontologies = self._config.get("ontologies", {})
        self._defaults = self._config.get("defaults", {})

        logger.info(
            "OntologyRouter initialized: %d datasets, %d ontologies",
            len(self._datasets), len(self._ontologies),
        )

    # == public API ========================================================

    @property
    def available_datasets(self) -> List[str]:
        return sorted(self._datasets.keys())

    @property
    def available_ontologies(self) -> List[str]:
        return sorted(self._ontologies.keys())

    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        name = dataset_name.lower()
        if name not in self._datasets:
            raise KeyError(
                f"Unknown dataset \'{dataset_name}\'. "
                f"Available: {self.available_datasets}"
            )
        return dict(self._datasets[name])

    def get_ontology_config(self, ontology_name: str) -> Dict[str, Any]:
        name = ontology_name.lower()
        if name not in self._ontologies:
            raise KeyError(
                f"Unknown ontology \'{ontology_name}\'. "
                f"Available: {self.available_ontologies}"
            )
        return dict(self._ontologies[name])

    def get_required_ontologies(
        self,
        dataset_name: str,
        *,
        ontology_override: Optional[str] = None,
    ) -> List[str]:
        """Return ontology list for *dataset_name*.

        Parameters
        ----------
        dataset_name : str
            Dataset name.
        ontology_override : str | None
            If provided, force this ontology regardless of dataset default.
        """
        if ontology_override:
            onto_name = ontology_override.lower()
            if onto_name not in self._ontologies:
                raise KeyError(
                    f"Unknown ontology '{ontology_override}'. "
                    f"Available: {self.available_ontologies}"
                )
            return [onto_name]

        ds_cfg = self.get_dataset_config(dataset_name)
        onto = ds_cfg.get("ontology")
        if onto:
            return [onto]
        # backward compat: legacy "ontologies" array
        return list(ds_cfg.get("ontologies", ["dto"]))

    def get_bridge_domain(
        self,
        dataset_name: str,
        *,
        ontology_override: Optional[str] = None,
    ) -> str:
        """Return bridge domain hint for *dataset_name*.

        Returns ``"chebi"`` or ``"anchor"``.
        """
        if ontology_override:
            # 현재 브릿지는 chebi 도메인만 별도 처리, 나머지는 anchor 사용
            return "chebi" if ontology_override.lower() == "chebi" else "anchor"

        ds_cfg = self.get_dataset_config(dataset_name)
        return ds_cfg.get("bridge_domain", "anchor")

    def route(
        self,
        dataset_name: str,
        *,
        extra_ontologies: Optional[List[str]] = None,
        ontology_override: Optional[str] = None,
    ) -> nx.DiGraph:
        """Load and return the single ontology graph for *dataset_name*.

        Parameters
        ----------
        dataset_name : str
        extra_ontologies : list[str] | None
            Additional ontologies to compose (rarely used).

        Returns
        -------
        nx.DiGraph
        """
        required = self.get_required_ontologies(
            dataset_name, ontology_override=ontology_override,
        )
        if extra_ontologies:
            for o in extra_ontologies:
                if o not in required:
                    required.append(o)

        logger.info(
            "Routing dataset '%s'  ->  ontology: %s",
            dataset_name, required,
        )

        composite = nx.DiGraph()
        for onto_name in required:
            t0 = time.time()
            sub = self._load_ontology(onto_name)
            elapsed = time.time() - t0

            for nid in sub.nodes:
                sub.nodes[nid]["source_ontology"] = onto_name

            composite = nx.compose(composite, sub)
            logger.info(
                "  Loaded '%s': %d nodes, %d edges (%.2fs)",
                onto_name,
                sub.number_of_nodes(), sub.number_of_edges(), elapsed,
            )

        logger.info(
            "Graph: %d nodes, %d edges",
            composite.number_of_nodes(), composite.number_of_edges(),
        )
        return composite

    def get_defaults(self) -> Dict[str, Any]:
        return dict(self._defaults)

    def print_routing_table(self) -> None:
        print("=" * 65)
        print(" Dataset -> Ontology Routing Table  (1:1)")
        print("=" * 65)
        for ds_name in self.available_datasets:
            ds_cfg = self._datasets[ds_name]
            onto = ds_cfg.get("ontology", ds_cfg.get("ontologies", ["?"])[0])
            domain = ds_cfg.get("bridge_domain", "anchor")
            profile = ds_cfg.get("heuristic_profile", "default")
            print(f"  {ds_name:12s} -> {onto:10s}  bridge={domain:8s}  [{profile}]")
        print("=" * 65)

    # == loading ============================================================

    def _load_ontology(self, onto_name: str) -> nx.DiGraph:
        if self.use_cache and onto_name in _ONTOLOGY_CACHE:
            logger.debug("Cache hit for \'{}\'", onto_name)
            return _ONTOLOGY_CACHE[onto_name].copy()

        onto_cfg = self.get_ontology_config(onto_name)
        owl_path = self.base_dir / onto_cfg["file"]

        if not owl_path.exists():
            logger.warning(
                "Ontology file not found: %s  ->  creating placeholder",
                owl_path,
            )
            G = self._create_placeholder_ontology(
                onto_name, onto_cfg.get("root_classes", []),
            )
            if self.use_cache:
                _ONTOLOGY_CACHE[onto_name] = G.copy()
            return G

        size_class = onto_cfg.get("size_class", "small")
        extract_sub = onto_cfg.get("extract_subgraph", False)
        max_triples = onto_cfg.get("max_triples", 0)

        if size_class == "large" and extract_sub:
            G = self._load_large_ontology(
                owl_path,
                max_triples=max_triples,
                root_classes=onto_cfg.get("root_classes", []),
                onto_name=onto_name,
            )
        else:
            builder = OWLGraphBuilder(str(owl_path))
            G = builder.build()

        if self.use_cache:
            _ONTOLOGY_CACHE[onto_name] = G.copy()
        return G

    def _load_large_ontology(
        self,
        owl_path: Path,
        *,
        max_triples: int = 300000,
        root_classes: Optional[list] = None,
        onto_name: str = "unknown",
    ) -> nx.DiGraph:
        """Load a large ontology with triple-count ceiling."""
        from rdflib import Graph as RDFGraph
        from rdflib import URIRef
        from rdflib.namespace import OWL, RDF, RDFS

        logger.info(
            "Loading large ontology: %s (max_triples=%d)",
            owl_path.name, max_triples,
        )

        G = nx.DiGraph()
        try:
            rdf = RDFGraph()
            rdf.parse(str(owl_path), format="xml")
            logger.info("  Parsed %d triples from %s", len(rdf), owl_path.name)

            added = 0
            for s, p, o in rdf.triples((None, RDFS.subClassOf, None)):
                if added >= max_triples:
                    break
                if isinstance(s, URIRef) and isinstance(o, URIRef):
                    src = self._short_label(s)
                    dst = self._short_label(o)
                    if src != dst:
                        G.add_node(src, label=src, uri=str(s))
                        G.add_node(dst, label=dst, uri=str(o))
                        G.add_edge(src, dst, predicate="subClassOf", pheromone=1.0)
                        added += 1

            logger.info(
                "  Extracted: %d nodes, %d edges",
                G.number_of_nodes(), G.number_of_edges(),
            )
        except Exception as e:
            logger.warning(
                "Failed to load %s: %s  ->  placeholder", owl_path.name, e,
            )
            G = self._create_placeholder_ontology(onto_name, root_classes)

        return G

    def _create_placeholder_ontology(
        self,
        name: str,
        root_classes: Optional[list] = None,
    ) -> nx.DiGraph:
        """Minimal placeholder when real ontology is unavailable."""
        G = nx.DiGraph()
        root = f"{name}:Thing"
        G.add_node(root, label=root, uri=f"urn:{name}:Thing")

        placeholder_hierarchies = {
            "chebi": [
                ("chemical_entity", "molecular_entity"),
                ("molecular_entity", "organic_molecule"),
                ("molecular_entity", "inorganic_molecule"),
                ("organic_molecule", "lipid"),
                ("organic_molecule", "amino_acid"),
                ("organic_molecule", "carbohydrate"),
                ("organic_molecule", "heterocyclic_compound"),
                ("organic_molecule", "aromatic_compound"),
            ],
            "go": [
                ("biological_process", "cellular_process"),
                ("biological_process", "metabolic_process"),
                ("biological_process", "viral_process"),
                ("viral_process", "viral_replication"),
                ("viral_process", "viral_entry"),
                ("cellular_process", "cell_death"),
                ("cellular_process", "signal_transduction"),
                ("metabolic_process", "biosynthetic_process"),
            ],
            "mesh": [
                ("system_organ_class", "nervous_system_disorders"),
                ("system_organ_class", "cardiac_disorders"),
                ("system_organ_class", "hepatobiliary_disorders"),
                ("system_organ_class", "gastrointestinal_disorders"),
                ("system_organ_class", "renal_disorders"),
                ("system_organ_class", "skin_disorders"),
                ("system_organ_class", "immune_disorders"),
                ("system_organ_class", "endocrine_disorders"),
                ("nervous_system_disorders", "high_level_group_term"),
                ("high_level_group_term", "preferred_term"),
            ],
            "bao": [
                ("bioassay", "biochemical_assay"),
                ("bioassay", "cell_based_assay"),
                ("bioassay", "organism_assay"),
                ("biochemical_assay", "enzyme_activity_assay"),
                ("biochemical_assay", "binding_assay"),
                ("cell_based_assay", "reporter_gene_assay"),
                ("cell_based_assay", "cell_viability_assay"),
            ],
            "dto": [
                ("drug_target", "protein_target"),
                ("drug_target", "nucleic_acid_target"),
                ("protein_target", "enzyme"),
                ("protein_target", "receptor"),
                ("protein_target", "ion_channel"),
                ("protein_target", "transporter"),
            ],
        }

        onto_type = None
        name_lower = name.lower()
        for key in placeholder_hierarchies:
            if key in name_lower:
                onto_type = key
                break

        if onto_type and onto_type in placeholder_hierarchies:
            for parent, child in placeholder_hierarchies[onto_type]:
                pid = f"{name}:{parent}"
                cid = f"{name}:{child}"
                G.add_node(pid, label=parent, uri=f"urn:{name}:{parent}")
                G.add_node(cid, label=child, uri=f"urn:{name}:{child}")
                G.add_edge(cid, pid, predicate="subClassOf", pheromone=1.0)
                if pid not in G.predecessors(root):
                    G.add_edge(pid, root, predicate="subClassOf", pheromone=1.0)

        if root_classes:
            for rc in root_classes:
                rc_id = f"{name}:{rc}"
                if rc_id not in G:
                    G.add_node(rc_id, label=rc, uri=f"urn:{name}:{rc}")
                    G.add_edge(rc_id, root, predicate="subClassOf", pheromone=1.0)

        logger.info(
            "  Placeholder '%s'  : %d nodes, %d edges",
            name, G.number_of_nodes(), G.number_of_edges(),
        )
        return G

    @staticmethod
    def _short_label(uri) -> str:
        s = str(uri)
        if "#" in s:
            return s.rsplit("#", 1)[-1]
        return s.rsplit("/", 1)[-1]
