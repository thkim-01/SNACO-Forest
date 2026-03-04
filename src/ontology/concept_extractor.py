"""
Extract dataset-specific concepts from OWL ontologies.
Maps molecules to relevant ontology concepts for improved DL-based refinement generation.
"""

from owlready2 import get_ontology
import logging
from typing import Dict, List, Set, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class OntologyConceptExtractor:
    """
    Extracts dataset-specific concepts from OWL files.
    Provides molecule-to-concept mappings for semantic refinement generation.
    """

    def __init__(self, ontology_paths: Dict[str, str]):
        """
        Args:
            ontology_paths: Dict mapping ontology name (e.g., 'GO', 'DTO', 'MESH', 'CHEBI', 'PATO', 'BAO')
                           to file paths (e.g., {'GO': 'path/to/go.owl', 'DTO': 'path/to/DTO.owl'})
        """
        self.ontologies = {}
        self.concept_cache = {}
        
        for onto_name, onto_path in ontology_paths.items():
            try:
                logger.info(f"Loading {onto_name} from {onto_path}")
                onto = get_ontology(onto_path).load()
                self.ontologies[onto_name] = onto
                logger.info(f"  ✓ {onto_name} loaded, {len(list(onto.classes()))} classes")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load {onto_name}: {e}")

    def get_dataset_relevant_concepts(self, dataset_name: str, ontology_names: List[str]) -> Dict[str, List]:
        """
        Extract dataset-relevant concepts from specified ontologies.
        
        Args:
            dataset_name: One of 'tox21', 'bace', 'clintox', 'hiv', 'sider', 'bbbp'
            ontology_names: List of ontology names, e.g., ['GO', 'DTO', 'CHEBI']
        
        Returns:
            Dict mapping ontology name to list of relevant concept IDs/names
        """
        if dataset_name in self.concept_cache:
            return self.concept_cache[dataset_name]
        
        concepts = {}
        
        # Dataset-specific concept extraction logic
        if dataset_name == 'tox21':
            # Tox21: nuclear/stress response assays → gene targets → GO processes
            concepts['BAO'] = self._extract_bao_assay_concepts()
            concepts['DTO'] = self._extract_dto_target_concepts(['AR', 'AhR', 'ESR1', 'AhR', 'HSF1'])
            concepts['GO'] = self._extract_go_concepts(['0003677', '0004871', '0004910', '0005634', '0008150', '0043401'])  # GO IDs relevant to TF/receptor
            concepts['CHEBI'] = self._extract_chebi_relevant_concepts()
            concepts['PATO'] = self._extract_pato_phenotype_concepts()
            
        elif dataset_name == 'bace':
            # BACE: β-secretase inhibition → aspartic protease → APP processing
            concepts['DTO'] = self._extract_dto_target_concepts(['protease', 'aspartic', 'inhibitor'])
            concepts['CHEBI'] = self._extract_chebi_relevant_concepts()
            concepts['GO'] = self._extract_go_concepts(['0004190', '0006508', '0016579'])  # protease activity, proteolysis, endosome
            
        elif dataset_name == 'clintox':
            # ClinTox: drug toxicity + FDA approval → clinical adverse effects + regulatory terms
            concepts['MESH'] = self._extract_mesh_adverse_effect_concepts()
            concepts['BAO'] = self._extract_bao_clinical_concepts()
            concepts['GO'] = self._extract_go_concepts(['0006952', '0009611', '0018887'])  # immune response, response to wounding
            concepts['CHEBI'] = self._extract_chebi_relevant_concepts()
            concepts['Thesaurus'] = self._extract_thesaurus_drug_concepts()
            
        elif dataset_name == 'hiv':
            # HIV: antiviral activity → viral processes + immune response
            concepts['BAO'] = self._extract_bao_antiviral_concepts()
            concepts['GO'] = self._extract_go_concepts(['0016032', '0019080', '0006952', '0045087'])  # viral life cycle, viral genome replication, immune response, innate immune
            concepts['CHEBI'] = self._extract_chebi_relevant_concepts()
            
        elif dataset_name == 'sider':
            # SIDER: side effects → organ systems + phenotypes
            concepts['MESH'] = self._extract_mesh_organ_system_concepts()
            concepts['Thesaurus'] = self._extract_thesaurus_side_effect_concepts()
            concepts['PATO'] = self._extract_pato_adverse_phenotype_concepts()
            concepts['CHEBI'] = self._extract_chebi_relevant_concepts()
            
        elif dataset_name == 'bbbp':
            # BBBP: blood-brain barrier penetration → lipophilicity + permeability
            concepts['CHEBI'] = self._extract_chebi_lipophilicity_concepts()
            
        self.concept_cache[dataset_name] = concepts
        return concepts

    def _extract_go_concepts(self, go_id_snippets: List[str]) -> List[str]:
        """Extract GO concepts based on ID snippets."""
        if 'GO' not in self.ontologies:
            return []
        
        onto = self.ontologies['GO']
        matching = []
        
        try:
            for cls in onto.classes():
                for snippet in go_id_snippets:
                    if snippet in str(cls.name):
                        matching.append(str(cls.name))
                        break
        except Exception as e:
            logger.warning(f"Error extracting GO concepts: {e}")
        
        return matching[:20]  # Limit to top 20 concepts

    def _extract_dto_target_concepts(self, keywords: List[str]) -> List[str]:
        """Extract DTO targets matching keywords."""
        if 'DTO' not in self.ontologies:
            return []
        
        onto = self.ontologies['DTO']
        matching = []
        
        try:
            for cls in onto.classes():
                class_name = str(cls.name).lower()
                for keyword in keywords:
                    if keyword.lower() in class_name:
                        matching.append(str(cls.name))
                        break
        except Exception as e:
            logger.warning(f"Error extracting DTO concepts: {e}")
        
        return matching[:15]

    def _extract_chebi_relevant_concepts(self) -> List[str]:
        """Extract CHEBI functional group and property concepts."""
        if 'CHEBI' not in self.ontologies:
            return []
        
        onto = self.ontologies['CHEBI']
        matching = []
        functional_groups = [
            'aromatic', 'alcohol', 'amine', 'carboxylic', 'ester', 'amide',
            'ketone', 'aldehyde', 'ether', 'thiol', 'sulfide', 'halide'
        ]
        
        try:
            for cls in onto.classes():
                class_name = str(cls.name).lower()
                for fg in functional_groups:
                    if fg in class_name:
                        matching.append(str(cls.name))
                        break
                if len(matching) >= 15:
                    break
        except Exception as e:
            logger.warning(f"Error extracting CHEBI concepts: {e}")
        
        return matching[:15]

    def _extract_mesh_adverse_effect_concepts(self) -> List[str]:
        """Extract MeSH adverse effect/pathological condition concepts."""
        if 'MESH' not in self.ontologies:
            return []
        
        onto = self.ontologies['MESH']
        matching = []
        adverse_keywords = [
            'toxicity', 'adverse', 'disease', 'disorder', 'patholog',
            'syndrome', 'injury', 'damage', 'necrosis', 'inflammation'
        ]
        
        try:
            for cls in onto.classes():
                class_name = str(cls.name).lower()
                for keyword in adverse_keywords:
                    if keyword in class_name:
                        matching.append(str(cls.name))
                        break
                if len(matching) >= 12:
                    break
        except Exception as e:
            logger.warning(f"Error extracting MeSH adverse concepts: {e}")
        
        return matching[:12]

    def _extract_mesh_organ_system_concepts(self) -> List[str]:
        """Extract MeSH organ system/anatomical concepts."""
        if 'MESH' not in self.ontologies:
            return []
        
        onto = self.ontologies['MESH']
        matching = []
        organ_keywords = [
            'liver', 'kidney', 'heart', 'brain', 'lung', 'blood',
            'skin', 'stomach', 'intestine', 'nervous', 'immune'
        ]
        
        try:
            for cls in onto.classes():
                class_name = str(cls.name).lower()
                for keyword in organ_keywords:
                    if keyword in class_name:
                        matching.append(str(cls.name))
                        break
                if len(matching) >= 10:
                    break
        except Exception as e:
            logger.warning(f"Error extracting MeSH organ concepts: {e}")
        
        return matching[:10]

    def _extract_pato_phenotype_concepts(self) -> List[str]:
        """Extract PATO phenotypic quality concepts."""
        if 'PATO' not in self.ontologies:
            return []
        
        onto = self.ontologies['PATO']
        matching = []
        pheno_keywords = ['quality', 'abnormal', 'morphology', 'increased', 'decreased']
        
        try:
            classes_list = list(onto.classes())[:100]  # Limit search
            for cls in classes_list:
                class_name = str(cls.name).lower()
                for keyword in pheno_keywords:
                    if keyword in class_name:
                        matching.append(str(cls.name))
                        break
                if len(matching) >= 10:
                    break
        except Exception as e:
            logger.warning(f"Error extracting PATO concepts: {e}")
        
        return matching[:10]

    def _extract_pato_adverse_phenotype_concepts(self) -> List[str]:
        """Extract PATO phenotypes related to adverse/pathological states."""
        return self._extract_pato_phenotype_concepts()

    def _extract_bao_assay_concepts(self) -> List[str]:
        """Extract BAO assay type and attribute concepts."""
        if 'BAO' not in self.ontologies:
            return []
        
        onto = self.ontologies['BAO']
        matching = []
        assay_keywords = [
            'assay', 'assay_type', 'screening', 'reporter', 'transcription',
            'nuclear', 'stress', 'phenotypic', 'detection'
        ]
        
        try:
            classes_list = list(onto.classes())[:500]  # BAO is large
            for cls in classes_list:
                class_name = str(cls.name).lower()
                for keyword in assay_keywords:
                    if keyword in class_name:
                        matching.append(str(cls.name))
                        break
                if len(matching) >= 12:
                    break
        except Exception as e:
            logger.warning(f"Error extracting BAO concepts: {e}")
        
        return matching[:12]

    def _extract_bao_clinical_concepts(self) -> List[str]:
        """Extract BAO clinical/toxicity related assay concepts."""
        if 'BAO' not in self.ontologies:
            return []
        
        onto = self.ontologies['BAO']
        matching = []
        clinical_keywords = ['clinical', 'toxicity', 'safety', 'efficacy', 'pharmacology']
        
        try:
            classes_list = list(onto.classes())[:300]
            for cls in classes_list:
                class_name = str(cls.name).lower()
                for keyword in clinical_keywords:
                    if keyword in class_name:
                        matching.append(str(cls.name))
                        break
                if len(matching) >= 8:
                    break
        except Exception as e:
            logger.warning(f"Error extracting BAO clinical concepts: {e}")
        
        return matching[:8]

    def _extract_bao_antiviral_concepts(self) -> List[str]:
        """Extract BAO antiviral assay concepts."""
        if 'BAO' not in self.ontologies:
            return []
        
        onto = self.ontologies['BAO']
        matching = []
        antiviral_keywords = ['antiviral', 'viral', 'infection', 'replication', 'inhibition']
        
        try:
            classes_list = list(onto.classes())[:300]
            for cls in classes_list:
                class_name = str(cls.name).lower()
                for keyword in antiviral_keywords:
                    if keyword in class_name:
                        matching.append(str(cls.name))
                        break
                if len(matching) >= 8:
                    break
        except Exception as e:
            logger.warning(f"Error extracting BAO antiviral concepts: {e}")
        
        return matching[:8]

    def _extract_chebi_lipophilicity_concepts(self) -> List[str]:
        """Extract CHEBI concepts related to lipophilicity and membrane permeability."""
        if 'CHEBI' not in self.ontologies:
            return []
        
        onto = self.ontologies['CHEBI']
        matching = []
        lipo_keywords = ['lipid', 'lipophil', 'hydrophob', 'amphiphil', 'nonpolar']
        
        try:
            classes_list = list(onto.classes())[:200]
            for cls in classes_list:
                class_name = str(cls.name).lower()
                for keyword in lipo_keywords:
                    if keyword in class_name:
                        matching.append(str(cls.name))
                        break
                if len(matching) >= 8:
                    break
        except Exception as e:
            logger.warning(f"Error extracting CHEBI lipophilicity concepts: {e}")
        
        return matching[:8]

    def _extract_thesaurus_drug_concepts(self) -> List[str]:
        """Extract Thesaurus drug-related concepts."""
        if 'Thesaurus' not in self.ontologies:
            return []
        
        onto = self.ontologies['Thesaurus']
        matching = []
        drug_keywords = ['drug', 'pharmaceutical', 'medication', 'compound', 'molecule']
        
        try:
            classes_list = list(onto.classes())[:200]
            for cls in classes_list:
                class_name = str(cls.name).lower()
                for keyword in drug_keywords:
                    if keyword in class_name:
                        matching.append(str(cls.name))
                        break
                if len(matching) >= 8:
                    break
        except Exception as e:
            logger.warning(f"Error extracting Thesaurus drug concepts: {e}")
        
        return matching[:8]

    def _extract_thesaurus_side_effect_concepts(self) -> List[str]:
        """Extract Thesaurus adverse effect/side effect concepts."""
        if 'Thesaurus' not in self.ontologies:
            return []
        
        onto = self.ontologies['Thesaurus']
        matching = []
        se_keywords = ['adverse', 'side effect', 'toxicity', 'complication', 'reaction']
        
        try:
            classes_list = list(onto.classes())[:200]
            for cls in classes_list:
                class_name = str(cls.name).lower()
                for keyword in se_keywords:
                    if keyword in class_name:
                        matching.append(str(cls.name))
                        break
                if len(matching) >= 8:
                    break
        except Exception as e:
            logger.warning(f"Error extracting Thesaurus side effect concepts: {e}")
        
        return matching[:8]

    def get_molecule_conceptattachments(self, 
                                       molecule_features: Dict,
                                       dataset_name: str,
                                       dataset_concepts: Dict[str, List]) -> Dict[str, List]:
        """
        Assign ontology concepts to a molecule based on its features and dataset context.
        
        Args:
            molecule_features: Dict with 'concepts', 'roles', 'properties' extracted from molecule
            dataset_name: Dataset name for context (e.g., 'tox21', 'bace')
            dataset_concepts: Result from get_dataset_relevant_concepts()
        
        Returns:
            Dict mapping ontology name to list of assigned concept names
        """
        attachments = defaultdict(list)
        
        # Map molecule features to ontology concepts heuristically
        mw = float(molecule_features.get('molecular_weight', 0.0) or 0.0)
        logp = float(molecule_features.get('logp', 0.0) or 0.0)
        hba = int(molecule_features.get('num_hba', 0) or 0)
        hbd = int(molecule_features.get('num_hbd', 0) or 0)
        fg_list = [str(x).lower() for x in (molecule_features.get('functional_groups', []) or [])]

        # Deterministic index selector to diversify per-molecule concept attachments
        selector = int(mw) + int(abs(logp) * 10) + hba + hbd + len(fg_list)

        def pick_variants(values: List[str], k: int = 3) -> List[str]:
            if not values:
                return []
            if len(values) <= k:
                return list(values)
            start = selector % max(1, len(values))
            out = []
            for i in range(k):
                out.append(values[(start + i) % len(values)])
            return out
        
        # CHEBI: Based on functional groups in molecule
        if 'CHEBI' in dataset_concepts:
            if any(fg in ' '.join(fg_list) for fg in ['alcohol', 'amine', 'aromatic', 'ester', 'carboxyl', 'amide']):
                attachments['CHEBI'].extend(pick_variants(dataset_concepts['CHEBI'], k=3))
        
        # DTO: Based on molecular properties
        if 'DTO' in dataset_concepts:
            if 100 < mw < 600:  # Drug-like MW range
                attachments['DTO'].extend(pick_variants(dataset_concepts['DTO'], k=3))
        
        # GO: Based on dataset context (all molecules in a task may share relevant GO terms)
        if 'GO' in dataset_concepts:
            if dataset_name in ('tox21', 'hiv', 'clintox'):
                attachments['GO'].extend(pick_variants(dataset_concepts['GO'], k=4))
        
        # BAO: For relevant datasets
        if 'BAO' in dataset_concepts:
            attachments['BAO'].extend(pick_variants(dataset_concepts['BAO'], k=3))
        
        # MESH: For clinical datasets
        if 'MESH' in dataset_concepts:
            attachments['MESH'].extend(pick_variants(dataset_concepts['MESH'], k=3))
        
        # PATO: For phenotype-related datasets
        if 'PATO' in dataset_concepts:
            attachments['PATO'].extend(pick_variants(dataset_concepts['PATO'], k=2))
        
        # Thesaurus: For drug approval/side effect datasets
        if 'Thesaurus' in dataset_concepts:
            attachments['Thesaurus'].extend(pick_variants(dataset_concepts['Thesaurus'], k=2))
        
        return dict(attachments)
