"""
Step 1: OWL(RDF/XML) → NetworkX DiGraph 변환기

rdflib로 OWL 파일을 파싱하고, owl:Class / rdfs:subClassOf / 사용자 정의
ObjectProperty 트리플만 필터링하여 경량 DiGraph를 생성한다.
모든 엣지에 pheromone=1.0 초기값을 부여한다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from rdflib import Graph as RDFGraph
from rdflib import Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS

logger = logging.getLogger(__name__)

# ── 기본 네임스페이스 ──────────────────────────────────────────
_FILTERED_PREDICATES: Set[URIRef] = {
    RDFS.subClassOf,
    OWL.equivalentClass,
    OWL.disjointWith,
    RDFS.domain,
    RDFS.range,
}

# owl:Class / rdfs:Class 판별용
_CLASS_TYPES: Set[URIRef] = {OWL.Class, RDFS.Class}

# rdflib 내장 네임스페이스 (사용자 정의 프로퍼티가 아닌 것)
_BUILTIN_NS_PREFIXES = (
    str(RDF),
    str(RDFS),
    str(OWL),
    "http://www.w3.org/2001/XMLSchema#",
    "http://www.w3.org/2002/07/owl#",
)


def _is_builtin_uri(uri: URIRef) -> bool:
    """rdflib 내장(RDF/RDFS/OWL/XSD) URI 여부 판별."""
    s = str(uri)
    return any(s.startswith(p) for p in _BUILTIN_NS_PREFIXES)


def _short_label(uri: URIRef) -> str:
    """URI → 읽기 쉬운 짧은 라벨 (fragment 또는 마지막 path segment)."""
    s = str(uri)
    if "#" in s:
        return s.rsplit("#", 1)[-1]
    return s.rsplit("/", 1)[-1]


class OWLGraphBuilder:
    """OWL(RDF/XML) 파일을 파싱하여 NetworkX DiGraph로 변환한다.

    Parameters
    ----------
    owl_path : str | Path
        OWL 파일 경로.
    default_pheromone : float
        엣지에 부여할 페로몬 초기값 (기본 1.0).
    include_object_properties : bool
        사용자 정의 ObjectProperty 기반 트리플도 포함할지 여부.

    Examples
    --------
    >>> builder = OWLGraphBuilder("ontology/DTO.owl")
    >>> G = builder.build()
    >>> builder.print_summary()
    """

    def __init__(
        self,
        owl_path: str | Path,
        default_pheromone: float = 1.0,
        include_object_properties: bool = True,
    ) -> None:
        self.owl_path = Path(owl_path)
        self.default_pheromone = default_pheromone
        self.include_object_properties = include_object_properties

        self._rdf: Optional[RDFGraph] = None
        self._graph: Optional[nx.DiGraph] = None

        # 클래스 URI → 짧은 라벨 매핑
        self._label_map: Dict[str, str] = {}
        # 사용자 정의 ObjectProperty URI 집합
        self._custom_obj_props: Set[URIRef] = set()

    # ── 공개 API ───────────────────────────────────────────────

    def build(self) -> nx.DiGraph:
        """OWL 파일을 파싱하고, 필터링된 트리플로 DiGraph를 생성한다.

        Returns
        -------
        nx.DiGraph
            변환된 유향 그래프. 노드에는 ``label``, ``uri`` 속성이,
            엣지에는 ``predicate``, ``pheromone`` 속성이 포함된다.
        """
        logger.info("Loading RDF/XML: %s", self.owl_path)
        self._rdf = RDFGraph()
        self._rdf.parse(str(self.owl_path), format=self._guess_format())
        logger.info(
            "Parsed %d raw triples from %s",
            len(self._rdf),
            self.owl_path.name,
        )

        self._collect_classes()
        if self.include_object_properties:
            self._collect_object_properties()

        self._graph = nx.DiGraph()
        self._add_class_triples()
        if self.include_object_properties:
            self._add_object_property_triples()

        logger.info(
            "Graph built — nodes: %d, edges: %d",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )
        return self._graph

    @property
    def graph(self) -> nx.DiGraph:
        """변환된 DiGraph를 반환한다 (build() 호출 후 사용)."""
        if self._graph is None:
            raise RuntimeError("build()를 먼저 호출하세요.")
        return self._graph

    def print_summary(self) -> None:
        """노드/엣지 수 요약을 출력한다."""
        G = self.graph
        print(f"[OWLGraphBuilder] Nodes: {G.number_of_nodes()}")
        print(f"[OWLGraphBuilder] Edges: {G.number_of_edges()}")

    def get_neighbors(self, node_id: str) -> List[Dict]:
        """특정 노드에 연결된 이웃 노드 정보를 반환한다.

        Parameters
        ----------
        node_id : str
            조회할 노드의 ID (짧은 라벨 또는 URI).

        Returns
        -------
        list[dict]
            각 이웃에 대한 ``{node, label, predicate, pheromone, direction}``
            딕셔너리 리스트.
        """
        G = self.graph
        if node_id not in G:
            logger.warning("Node '%s' not found in graph.", node_id)
            return []

        neighbors: List[Dict] = []

        # 나가는 엣지 (successors)
        for succ in G.successors(node_id):
            edata = G.edges[node_id, succ]
            neighbors.append(
                {
                    "node": succ,
                    "label": G.nodes[succ].get("label", succ),
                    "predicate": edata.get("predicate", ""),
                    "pheromone": edata.get("pheromone", self.default_pheromone),
                    "direction": "outgoing",
                }
            )

        # 들어오는 엣지 (predecessors)
        for pred in G.predecessors(node_id):
            edata = G.edges[pred, node_id]
            neighbors.append(
                {
                    "node": pred,
                    "label": G.nodes[pred].get("label", pred),
                    "predicate": edata.get("predicate", ""),
                    "pheromone": edata.get("pheromone", self.default_pheromone),
                    "direction": "incoming",
                }
            )

        return neighbors

    def get_label_map(self) -> Dict[str, str]:
        """노드 ID → 짧은 라벨 매핑 딕셔너리."""
        return dict(self._label_map)

    # ── 내부 구현 ──────────────────────────────────────────────

    def _guess_format(self) -> str:
        """확장자로부터 rdflib 파싱 포맷을 추론한다."""
        suffix = self.owl_path.suffix.lower()
        fmt_map = {
            ".owl": "xml",
            ".rdf": "xml",
            ".xrdf": "xml",
            ".xml": "xml",
            ".ttl": "turtle",
            ".nt": "nt",
            ".n3": "n3",
            ".jsonld": "json-ld",
        }
        return fmt_map.get(suffix, "xml")

    def _collect_classes(self) -> None:
        """owl:Class / rdfs:Class 로 선언된 URI를 수집한다."""
        assert self._rdf is not None
        for cls_type in _CLASS_TYPES:
            for s in self._rdf.subjects(RDF.type, cls_type):
                if isinstance(s, URIRef):
                    label = self._resolve_label(s)
                    self._label_map[label] = label

    def _collect_object_properties(self) -> None:
        """사용자 정의 ObjectProperty URI를 수집한다."""
        assert self._rdf is not None
        for s in self._rdf.subjects(RDF.type, OWL.ObjectProperty):
            if isinstance(s, URIRef) and not _is_builtin_uri(s):
                self._custom_obj_props.add(s)
        logger.info("Custom ObjectProperties found: %d", len(self._custom_obj_props))

    def _resolve_label(self, uri: URIRef) -> str:
        """URI의 rdfs:label이 있으면 사용하고, 없으면 fragment를 사용한다."""
        assert self._rdf is not None
        for lbl in self._rdf.objects(uri, RDFS.label):
            return str(lbl)
        return _short_label(uri)

    def _ensure_node(self, uri: URIRef) -> str:
        """그래프에 노드를 추가하고 ID(라벨)를 반환한다."""
        assert self._graph is not None
        label = self._resolve_label(uri)
        if label not in self._graph:
            self._graph.add_node(label, label=label, uri=str(uri))
        return label

    def _add_class_triples(self) -> None:
        """owl:Class 간의 rdfs:subClassOf 등 구조적 관계를 엣지로 추가한다."""
        assert self._rdf is not None and self._graph is not None
        added = 0
        for pred_uri in _FILTERED_PREDICATES:
            for s, _, o in self._rdf.triples((None, pred_uri, None)):
                if not (isinstance(s, URIRef) and isinstance(o, URIRef)):
                    continue
                src = self._ensure_node(s)
                dst = self._ensure_node(o)
                if src == dst:
                    continue
                self._graph.add_edge(
                    src,
                    dst,
                    predicate=_short_label(pred_uri),
                    pheromone=self.default_pheromone,
                )
                added += 1
        logger.info("Class-level edges added: %d", added)

    def _add_object_property_triples(self) -> None:
        """사용자 정의 ObjectProperty 기반 트리플을 엣지로 추가한다."""
        assert self._rdf is not None and self._graph is not None
        added = 0
        for prop_uri in self._custom_obj_props:
            for s, _, o in self._rdf.triples((None, prop_uri, None)):
                if not (isinstance(s, URIRef) and isinstance(o, URIRef)):
                    continue
                src = self._ensure_node(s)
                dst = self._ensure_node(o)
                if src == dst:
                    continue
                self._graph.add_edge(
                    src,
                    dst,
                    predicate=_short_label(prop_uri),
                    pheromone=self.default_pheromone,
                )
                added += 1
        logger.info("ObjectProperty edges added: %d", added)


# ── CLI 실행 ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    owl_file = sys.argv[1] if len(sys.argv) > 1 else "ontology/DTO.owl"
    builder = OWLGraphBuilder(owl_file)
    G = builder.build()
    builder.print_summary()

    # 샘플 노드 이웃 조회
    sample_nodes = list(G.nodes)[:3]
    for node in sample_nodes:
        nbrs = builder.get_neighbors(node)
        print(f"\n[{node}] neighbors ({len(nbrs)}):")
        for n in nbrs[:5]:
            print(f"  → {n['label']} (pred={n['predicate']}, "
                  f"pheromone={n['pheromone']:.2f}, dir={n['direction']})")
