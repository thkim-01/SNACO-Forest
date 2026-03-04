"""
Molecule Ontology: 화학 온톨로지 구조 정의
"""
from typing import List, Dict, Set, Optional
from owlready2 import *
import os
import time
import threading
from pathlib import Path


class MoleculeOntology:
    """화학 분자 온톨로지를 관리하는 클래스"""
    
    def __init__(
        self,
        ontology_path: str = "ontology/DTO.xrdf",
        base_dto_path: Optional[str] = None,
    ):
        """
        Args:
            ontology_path: 작업용/저장용 온톨로지 파일 경로.
                - 파일이 이미 존재하면 해당 파일을 로드합니다.
                - 파일이 없으면 base DTO(기본: DTO.owl 또는 DTO.xrdf)를 로드한 뒤
                  확장합니다.
            base_dto_path: 신규 온톨로지 생성 시 기반으로 사용할 DTO 파일 경로(선택).
        """

        # Target ontology path (workspace-specific ontology for the dataset)
        self.ontology_path = ontology_path
        self.base_dto_path = (
            base_dto_path or self._resolve_default_base_dto_path()
        )
        self.onto = None
        self._load_and_enrich_ontology()

    @staticmethod
    def _resolve_default_base_dto_path() -> Optional[str]:
        """Pick a local DTO source file.

        In practice, `DTO.xrdf` has been the most robust to parse locally.
        `DTO.owl` may include imports / constructs that can fail to load
        depending on Owlready2 version or environment.
        """
        candidates = [
            os.path.join("ontology", "DTO.xrdf"),
            os.path.join("ontology", "DTO.owl"),
            os.path.join("ontology", "DTO.xml"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None
    
    def _load_and_enrich_ontology(self):
        """Load DTO and inject Chemical Ontology structure"""
        def _log(msg: str):
            print(f"[OntologyLoader] {msg}", flush=True)

        def _load_with_progress(path_norm: str, label: str):
            """Load ontology while printing heartbeat logs during blocking parse."""
            started_at = time.time()
            stop_event = threading.Event()

            def _heartbeat():
                while not stop_event.wait(5.0):
                    elapsed = time.time() - started_at
                    _log(f"{label} still parsing... elapsed={elapsed:.1f}s")

            hb = threading.Thread(target=_heartbeat, daemon=True)
            hb.start()
            try:
                onto = get_ontology(path_norm).load(only_local=True)
                return onto, started_at
            finally:
                stop_event.set()
                hb.join(timeout=0.2)
        
        # 1. Load an ontology
        # - If ontology_path already exists (previously generated dataset ontology), load it.
        # - Otherwise, load base DTO.owl (or DTO.xrdf) and extend it.
        loaded_from = None
        if os.path.exists(self.ontology_path):
            loaded_from = self.ontology_path
        elif self.base_dto_path and os.path.exists(self.base_dto_path):
            loaded_from = self.base_dto_path

        if loaded_from:
            _log(f"Loading base ontology from {loaded_from}")
            try:
                # Avoid slow / brittle remote owl:imports downloads.
                # If imported ontologies exist as local files in `ontology/`,
                # Owlready2 will pick them up from onto_path.
                onto_dir = os.path.abspath("ontology")
                if onto_dir not in onto_path:
                    onto_path.append(onto_dir)
                    _log(f"Registered local import path: {onto_dir}")

                loaded_norm = Path(loaded_from).as_posix()
                _log(f"Parsing ontology file (only_local=True): {loaded_norm}")
                self.onto, started_at = _load_with_progress(loaded_norm, "Primary ontology")
                _log(
                    f"Ontology parse complete in {time.time() - started_at:.2f}s "
                    f"(classes={len(list(self.onto.classes()))})"
                )
            except Exception as e:
                # Try other local DTO candidates before falling back to a blank ontology.
                _log(f"Primary ontology load failed for {loaded_from}: {e}")

                fallback_candidates = [
                    os.path.join("ontology", "DTO.xrdf"),
                    os.path.join("ontology", "DTO.owl"),
                    os.path.join("ontology", "DTO.xml"),
                ]
                fallback_loaded = False
                for fb in fallback_candidates:
                    if (fb != loaded_from) and os.path.exists(fb):
                        try:
                            _log(f"Trying fallback ontology: {fb}")
                            fb_norm = Path(fb).as_posix()
                            self.onto, started_at = _load_with_progress(fb_norm, f"Fallback ontology ({fb})")
                            _log(
                                f"Fallback load complete in {time.time() - started_at:.2f}s "
                                f"(classes={len(list(self.onto.classes()))})"
                            )
                            fallback_loaded = True
                            break
                        except Exception as e2:
                            _log(f"Fallback load failed for {fb}: {e2}")

                if not fallback_loaded:
                    _log("All local ontology candidates failed. Creating new empty ontology base")
                    self.onto = get_ontology(
                        "http://www.semanticweb.org/molecule/ontology"
                    )
        else:
            _log("No ontology file found. Creating new ontology base")
            self.onto = get_ontology("http://www.semanticweb.org/molecule/ontology")
            
        # 2. Enrich with Chemical Classes
        _log("Starting ontology enrichment: classes/properties injection")
        with self.onto:
            # Check if classes already exist to avoid duplication if re-loading
            if not self.onto['Molecule']:
                class Molecule(Thing):
                    """Center Class: Molecule (Enriched into DTO)"""
                    pass
            else:
                Molecule = self.onto.Molecule

            # Helper to safely creation
            def get_or_create(name, parent):
                c = self.onto[name]
                if not c:
                    with self.onto:
                        return type(name, (parent,), {})
                return c
            
            self.AromaticMolecule = get_or_create('AromaticMolecule', Molecule)
            self.NonAromaticMolecule = get_or_create('NonAromaticMolecule', Molecule)
            
            self.Substructure = get_or_create('Substructure', Thing)
            self.FunctionalGroup = get_or_create('FunctionalGroup', self.Substructure)
            self.RingSystem = get_or_create('RingSystem', self.Substructure)
            
            # Functionals
            groups = ['Alcohol', 'Amine', 'Carboxyl', 'Carbonyl', 'Ether', 'Ester', 'Amide', 'Nitro', 'Halogen']
            for g in groups:
                 setattr(self, g, get_or_create(g, self.FunctionalGroup))

            # v2 확장 작용기/구조 알럿 클래스
            extended_fg = [
                'Phenol', 'Epoxide', 'MichaelAcceptor', 'AromaticAmine',
                'AromaticNitro', 'Quinone', 'Thiophene', 'Imidazole',
                'Triazole', 'Amidine', 'Guanidine', 'Acylguanidine',
                'Hydroxyethylamine', 'Iminohydantoin', 'QuaternaryN',
                'Nucleoside', 'MetalChelator', 'Carboxylicacid',
                'Sulfonamide', 'Steroid', 'Alkylhalide', 'Acylhalide',
                'StrongElectrophile_Mustard', 'Hydrazine', 'Sulfur',
            ]
            for g in extended_fg:
                setattr(self, g, get_or_create(g, self.FunctionalGroup))
            
            # Rings
            self.BenzeneRing = get_or_create('BenzeneRing', self.RingSystem)
            self.Heterocycle = get_or_create('Heterocycle', self.RingSystem)

            # GO terms (for pathway/toxicity annotations)
            self.GOTerm = get_or_create('GOTerm', Thing)
            self.MeshTerm = get_or_create('MeshTerm', Thing)
            self.AnnotationTerm = get_or_create('AnnotationTerm', Thing)

            # Dataset-specific relation target classes
            self.BindingCore = get_or_create('BindingCore', Thing)
            self.ViralLifecycleStep = get_or_create('ViralLifecycleStep', Thing)
            self.AOP = get_or_create('AOP', Thing)
            self.AdverseLLT = get_or_create('AdverseLLT', Thing)
            self.AdversePT = get_or_create('AdversePT', Thing)
            self.AdverseSOC = get_or_create('AdverseSOC', Thing)

            # v4: BBBP - Transporter / physicochemical risk
            self.Transporter = get_or_create('Transporter', Thing)
            self.PhysicochemicalRisk = get_or_create('PhysicochemicalRisk', Thing)

            # v4: BACE - catalytic residue & selectivity
            self.CatalyticResidue = get_or_create('CatalyticResidue', Thing)
            self.SelectivityClass = get_or_create('SelectivityClass', Thing)

            # v4: ClinTox - EDT / Cramer / reactive toxicophore
            self.EDTClass = get_or_create('EDTClass', Thing)
            self.CramerClass = get_or_create('CramerClass', Thing)
            self.ReactiveToxicophore = get_or_create('ReactiveToxicophore', Thing)

            # v4: HIV - viral protein targets
            self.ViralProtein = get_or_create('ViralProtein', Thing)

            # v4: Tox21 - biological pathway
            self.BiologicalPathway = get_or_create('BiologicalPathway', Thing)

            # Properties
            # We use 'search' or just define. Ideally unique names.
            with self.onto:
                class hasSubstructure(Molecule >> self.Substructure): pass
                class hasFunctionalGroupRel(hasSubstructure): range = [self.FunctionalGroup]
                class hasRingSystem(hasSubstructure): range = [self.RingSystem]
                class hasGOTerm(Molecule >> self.GOTerm): pass
                class hasMeshTerm(Molecule >> self.MeshTerm): pass
                class hasAnnotationTerm(Molecule >> self.AnnotationTerm): pass
                class hasBindingCore(Molecule >> self.BindingCore): pass
                class inhibitsStep(Molecule >> self.ViralLifecycleStep): pass
                class hasAOP(Molecule >> self.AOP): pass
                class hasAdverseLLT(Molecule >> self.AdverseLLT): pass
                class hasAdversePT(Molecule >> self.AdversePT): pass
                class hasAdverseSOC(Molecule >> self.AdverseSOC): pass
                class lltBelongsToPT(self.AdverseLLT >> self.AdversePT): pass
                class ptBelongsToSOC(self.AdversePT >> self.AdverseSOC): pass
                # v4: BBBP properties
                class is_substrate_of(Molecule >> self.Transporter): pass
                class has_physicochemical_risk(Molecule >> self.PhysicochemicalRisk): pass
                # v4: BACE properties
                class interactsWithResidue(Molecule >> self.CatalyticResidue): pass
                # v4: ClinTox properties
                class isMemberOfEDTClass(Molecule >> self.EDTClass): pass
                class isMemberOfCramerClass(Molecule >> self.CramerClass): pass
                class hasReactiveToxicophore(Molecule >> self.ReactiveToxicophore): pass
                # v4: HIV properties
                class targetsProtein(Molecule >> self.ViralProtein): pass
                # v4: Tox21 properties
                class is_involved_in_pathway(Molecule >> self.BiologicalPathway): pass
                # v4: SIDER primary SOC mapping
                class mapsToPrimarySOC(self.AdversePT >> self.AdverseSOC): pass
                
                # Data Properties
                if not self.onto['hasMolecularWeight']:
                    class hasMolecularWeight(Molecule >> float): pass
                    class hasNumAtoms(Molecule >> int): pass
                    class hasNumHeavyAtoms(Molecule >> int): pass
                    class hasNumRotatableBonds(Molecule >> int): pass
                    class hasNumHBA(Molecule >> int): pass
                    class hasNumHBD(Molecule >> int): pass
                    class hasNumRings(Molecule >> int): pass
                    class hasNumAromaticRings(Molecule >> int): pass
                    class hasAromaticity(Molecule >> bool): pass
                    class hasLogP(Molecule >> float): pass
                    class hasTPSA(Molecule >> float): pass
                    class obeysLipinski(Molecule >> bool): pass
                    class hasMWCategory(Molecule >> str): pass
                    class hasLogPCategory(Molecule >> str): pass
                    class hasTPSACategory(Molecule >> str): pass
                    class hasLabel(Molecule >> int): pass

                # v2 확장 Data Properties
                if not self.onto['hasMolarRefractivity']:
                    class hasMolarRefractivity(Molecule >> float):
                        """몰 굴절률 (Crippen MR)"""
                        pass
                    class hasNPlusOCount(Molecule >> int):
                        """질소+산소 원자 합계"""
                        pass
                    class hasFormalCharge(Molecule >> int):
                        """형식 전하 (Formal Charge)"""
                        pass
                    class hasNumHeteroatoms(Molecule >> int):
                        """이종 원자 수"""
                        pass
                    class hasFsp3(Molecule >> float):
                        """sp3 탄소 분율 (Fraction CSP3)"""
                        pass

        # Expose
        self.Molecule = Molecule
        self.AromaticMolecule = self.AromaticMolecule
        self.NonAromaticMolecule = self.NonAromaticMolecule
        self.Substructure = self.Substructure
        self.FunctionalGroup = self.FunctionalGroup
        self.RingSystem = self.RingSystem
        
        self.hasSubstructure = self.onto.hasSubstructure
        self.hasFunctionalGroupRel = self.onto.hasFunctionalGroupRel
        self.hasRingSystem = self.onto.hasRingSystem
        self.hasGOTerm = self.onto.hasGOTerm
        self.hasMeshTerm = self.onto.hasMeshTerm
        self.hasAnnotationTerm = self.onto.hasAnnotationTerm
        self.hasBindingCore = self.onto.hasBindingCore
        self.inhibitsStep = self.onto.inhibitsStep
        self.hasAOP = self.onto.hasAOP
        self.hasAdverseLLT = self.onto.hasAdverseLLT
        self.hasAdversePT = self.onto.hasAdversePT
        self.hasAdverseSOC = self.onto.hasAdverseSOC
        self.lltBelongsToPT = self.onto.lltBelongsToPT
        self.ptBelongsToSOC = self.onto.ptBelongsToSOC
        # v4 properties
        self.is_substrate_of = self.onto.is_substrate_of
        self.has_physicochemical_risk = self.onto.has_physicochemical_risk
        self.interactsWithResidue = self.onto.interactsWithResidue
        self.isMemberOfEDTClass = self.onto.isMemberOfEDTClass
        self.isMemberOfCramerClass = self.onto.isMemberOfCramerClass
        self.hasReactiveToxicophore = self.onto.hasReactiveToxicophore
        self.targetsProtein = self.onto.targetsProtein
        self.is_involved_in_pathway = self.onto.is_involved_in_pathway
        self.mapsToPrimarySOC = self.onto.mapsToPrimarySOC
        
        # Expose subclasses
        self.Alcohol = self.onto.Alcohol
        self.Amine = self.onto.Amine
        self.Carboxyl = self.onto.Carboxyl
        self.Carbonyl = self.onto.Carbonyl
        self.Ether = self.onto.Ether
        self.Ester = self.onto.Ester
        self.Amide = self.onto.Amide
        self.Nitro = self.onto.Nitro
        self.Halogen = self.onto.Halogen
        self.BenzeneRing = self.onto.BenzeneRing
        self.Heterocycle = self.onto.Heterocycle
        self.GOTerm = self.onto.GOTerm
        self.MeshTerm = self.onto.MeshTerm
        self.AnnotationTerm = self.onto.AnnotationTerm
        _log("Ontology enrichment completed")
    
    def add_molecule_instance(self, mol_id: str, features: Dict, label: int):
        """분자 인스턴스를 온톨로지에 추가"""
        with self.onto:
            mol_instance = self.Molecule(mol_id)
            
            # Data properties 설정
            mol_instance.hasMolecularWeight = [features['molecular_weight']]
            mol_instance.hasNumAtoms = [features['num_atoms']]
            mol_instance.hasNumHeavyAtoms = [features['num_heavy_atoms']]
            mol_instance.hasNumRotatableBonds = [features['num_rotatable_bonds']]
            mol_instance.hasNumHBA = [features['num_hba']]
            mol_instance.hasNumHBD = [features['num_hbd']]
            mol_instance.hasNumRings = [features['num_rings']]
            mol_instance.hasNumAromaticRings = [features['num_aromatic_rings']]
            mol_instance.hasAromaticity = [features['has_aromatic']]
            mol_instance.hasLogP = [features['logp']]
            mol_instance.hasTPSA = [features['tpsa']]
            mol_instance.obeysLipinski = [features['obeys_lipinski']]
            mol_instance.hasMWCategory = [features['mw_category']]
            mol_instance.hasLogPCategory = [features['logp_category']]
            mol_instance.hasTPSACategory = [features['tpsa_category']]

            # v2 확장 Data Properties
            mol_instance.hasMolarRefractivity = [features.get('molar_refractivity', 0.0)]
            mol_instance.hasNPlusOCount = [features.get('n_plus_o_count', 0)]
            mol_instance.hasFormalCharge = [features.get('formal_charge', 0)]
            mol_instance.hasNumHeteroatoms = [features.get('num_heteroatoms', 0)]
            mol_instance.hasFsp3 = [features.get('fsp3', 0.0)]
            
            
            # --- Populate Object Properties for True SDT ---
            # Map string features to Ontology Classes (기본 + v2 구조 알럿)
            fg_map = {
                'Alcohol': self.Alcohol,
                'Amine': self.Amine,
                'Carboxyl': self.Carboxyl,
                'Carbonyl': self.Carbonyl,
                'Ether': self.Ether,
                'Ester': self.Ester,
                'Amide': self.Amide,
                'Nitro': self.Nitro,
                'Halogen': self.Halogen,
                'Benzene': self.BenzeneRing,
                # v2 확장 작용기/구조 알럿
                'Phenol': self.onto.Phenol,
                'Epoxide': self.onto.Epoxide,
                'MichaelAcceptor': self.onto.MichaelAcceptor,
                'AromaticAmine': self.onto.AromaticAmine,
                'AromaticNitro': self.onto.AromaticNitro,
                'Quinone': self.onto.Quinone,
                'Thiophene': self.onto.Thiophene,
                'Imidazole': self.onto.Imidazole,
                'Triazole': self.onto.Triazole,
                'Amidine': self.onto.Amidine,
                'Guanidine': self.onto.Guanidine,
                'Acylguanidine': self.onto.Acylguanidine,
                'Hydroxyethylamine': self.onto.Hydroxyethylamine,
                'Sulfur': self.onto.Sulfur,
                'Alkylhalide': self.onto.Alkylhalide,
                'Acylhalide': self.onto.Acylhalide,
                'QuaternaryN': self.onto.QuaternaryN,
                'Sulfonamide': self.onto.Sulfonamide,
                'Hydrazine': self.onto.Hydrazine,
            }

            # 1) 기본 functional_groups 처리
            for fg_name, fg_class in fg_map.items():
                if fg_name in features.get('functional_groups', []):
                    if fg_class:
                        fg_instance = fg_class()
                        mol_instance.hasFunctionalGroupRel.append(fg_instance)

            # 2) v2 structural_alerts 처리 (functional_groups에 이미 포함된 것은 스킵)
            already_added = set(features.get('functional_groups', []))
            for alert_name in features.get('structural_alerts', []):
                if alert_name in already_added:
                    continue
                alert_class = fg_map.get(alert_name) or getattr(self.onto, alert_name, None)
                if alert_class:
                    alert_instance = alert_class()
                    mol_instance.hasFunctionalGroupRel.append(alert_instance)
                    already_added.add(alert_name)

            # 3) BACE core relation: hasBindingCore
            for core_name in features.get('binding_cores', []) or []:
                core_id = str(core_name).strip()
                if not core_id:
                    continue
                class_name = f"BindingCore_{core_id.replace(':', '_')}"
                core_class = self.onto[class_name]
                if not core_class:
                    with self.onto:
                        core_class = type(class_name, (self.BindingCore,), {})
                core_instance = core_class()
                mol_instance.hasBindingCore.append(core_instance)

            # 4) HIV mechanism relation: inhibitsStep
            for step_name in features.get('inhibits_steps', []) or []:
                step_id = str(step_name).strip()
                if not step_id:
                    continue
                class_name = f"ViralStep_{step_id.replace(':', '_').replace(' ', '_')}"
                step_class = self.onto[class_name]
                if not step_class:
                    with self.onto:
                        step_class = type(class_name, (self.ViralLifecycleStep,), {})
                step_instance = step_class()
                mol_instance.inhibitsStep.append(step_instance)

            # 5) Tox21 relation: hasAOP
            for aop_name in features.get('aop_terms', []) or []:
                aop_id = str(aop_name).strip()
                if not aop_id:
                    continue
                class_name = f"AOP_{aop_id.replace(':', '_').replace(' ', '_').replace('-', '_')}"
                aop_class = self.onto[class_name]
                if not aop_class:
                    with self.onto:
                        aop_class = type(class_name, (self.AOP,), {})
                aop_instance = aop_class()
                mol_instance.hasAOP.append(aop_instance)

            # 6) SIDER hierarchy relation: LLT -> PT -> SOC
            llt_terms = features.get('sider_llt_terms', []) or []
            pt_terms = features.get('sider_pt_terms', []) or []
            soc_terms = features.get('sider_soc_terms', []) or []

            pt_instances = {}
            soc_instances = {}

            for soc_name in soc_terms:
                soc_id = str(soc_name).strip()
                if not soc_id:
                    continue
                class_name = f"AdverseSOC_{soc_id.replace(':', '_').replace(' ', '_').replace('-', '_')}"
                soc_class = self.onto[class_name]
                if not soc_class:
                    with self.onto:
                        soc_class = type(class_name, (self.AdverseSOC,), {})
                soc_inst = soc_class()
                soc_instances[soc_id] = soc_inst
                mol_instance.hasAdverseSOC.append(soc_inst)

            for pt_name in pt_terms:
                pt_id = str(pt_name).strip()
                if not pt_id:
                    continue
                class_name = f"AdversePT_{pt_id.replace(':', '_').replace(' ', '_').replace('-', '_')}"
                pt_class = self.onto[class_name]
                if not pt_class:
                    with self.onto:
                        pt_class = type(class_name, (self.AdversePT,), {})
                pt_inst = pt_class()
                pt_instances[pt_id] = pt_inst
                mol_instance.hasAdversePT.append(pt_inst)

                # PT -> SOC roll-up relation
                for soc_inst in soc_instances.values():
                    pt_inst.ptBelongsToSOC.append(soc_inst)

            for llt_name in llt_terms:
                llt_id = str(llt_name).strip()
                if not llt_id:
                    continue
                class_name = f"AdverseLLT_{llt_id.replace(':', '_').replace(' ', '_').replace('-', '_')}"
                llt_class = self.onto[class_name]
                if not llt_class:
                    with self.onto:
                        llt_class = type(class_name, (self.AdverseLLT,), {})
                llt_inst = llt_class()
                mol_instance.hasAdverseLLT.append(llt_inst)

                # LLT -> PT roll-up relation
                for pt_inst in pt_instances.values():
                    llt_inst.lltBelongsToPT.append(pt_inst)

            # 6b) SIDER: mapsToPrimarySOC (multi-axiality priority)
            primary_soc_name = features.get('primary_soc', '')
            if primary_soc_name:
                soc_id = str(primary_soc_name).strip()
                cls_name = f"AdverseSOC_{soc_id.replace(':', '_').replace(' ', '_').replace('-', '_')}"
                pri_soc_class = self.onto[cls_name]
                if not pri_soc_class:
                    with self.onto:
                        pri_soc_class = type(cls_name, (self.AdverseSOC,), {})
                pri_soc_inst = pri_soc_class()
                for pt_inst in pt_instances.values():
                    pt_inst.mapsToPrimarySOC.append(pri_soc_inst)

            # 7) BBBP: transporter substrate relations
            for sub_name in features.get('transporter_substrates', []) or []:
                sub_id = str(sub_name).strip()
                if not sub_id:
                    continue
                cls_name = f"Transporter_{sub_id}"
                sub_class = self.onto[cls_name]
                if not sub_class:
                    with self.onto:
                        sub_class = type(cls_name, (self.Transporter,), {})
                mol_instance.is_substrate_of.append(sub_class())

            # 8) BBBP: physicochemical risk relations
            for risk_name in features.get('physicochemical_risks', []) or []:
                risk_id = str(risk_name).strip()
                if not risk_id:
                    continue
                cls_name = f"PhysicoRisk_{risk_id}"
                risk_class = self.onto[cls_name]
                if not risk_class:
                    with self.onto:
                        risk_class = type(cls_name, (self.PhysicochemicalRisk,), {})
                mol_instance.has_physicochemical_risk.append(risk_class())

            # 9) BACE: interacts with catalytic residues
            for res_name in features.get('interacts_residues', []) or []:
                res_id = str(res_name).strip()
                if not res_id:
                    continue
                cls_name = f"Residue_{res_id}"
                res_class = self.onto[cls_name]
                if not res_class:
                    with self.onto:
                        res_class = type(cls_name, (self.CatalyticResidue,), {})
                mol_instance.interactsWithResidue.append(res_class())

            # 10) BACE: selectivity classification
            sel_class_name = features.get('selectivity_class', '')
            if sel_class_name:
                sel_cls = self.onto[sel_class_name]
                if not sel_cls:
                    with self.onto:
                        sel_cls = type(sel_class_name, (self.SelectivityClass,), {})
                mol_instance.is_a.append(sel_cls)

            # 11) ClinTox: EDT class membership
            edt_name = features.get('edt_class', '')
            if edt_name:
                edt_cls = self.onto[edt_name]
                if not edt_cls:
                    with self.onto:
                        edt_cls = type(edt_name, (self.EDTClass,), {})
                mol_instance.isMemberOfEDTClass.append(edt_cls())

            # 12) ClinTox: Cramer class membership
            cramer_name = features.get('cramer_class', '')
            if cramer_name:
                cramer_cls = self.onto[cramer_name]
                if not cramer_cls:
                    with self.onto:
                        cramer_cls = type(cramer_name, (self.CramerClass,), {})
                mol_instance.isMemberOfCramerClass.append(cramer_cls())

            # 13) ClinTox: reactive toxicophore relations
            for tox_name in features.get('reactive_toxicophores', []) or []:
                tox_id = str(tox_name).strip()
                if not tox_id:
                    continue
                cls_name = f"Toxicophore_{tox_id}"
                tox_class = self.onto[cls_name]
                if not tox_class:
                    with self.onto:
                        tox_class = type(cls_name, (self.ReactiveToxicophore,), {})
                mol_instance.hasReactiveToxicophore.append(tox_class())

            # 14) HIV: targets protein (PR, RT, IN)
            for prot_name in features.get('targets_proteins', []) or []:
                prot_id = str(prot_name).strip()
                if not prot_id:
                    continue
                cls_name = f"ViralProtein_{prot_id}"
                prot_class = self.onto[cls_name]
                if not prot_class:
                    with self.onto:
                        prot_class = type(cls_name, (self.ViralProtein,), {})
                mol_instance.targetsProtein.append(prot_class())

            # 15) Tox21: biological pathway involvement
            for pw_name in features.get('pathway_terms', []) or []:
                pw_id = str(pw_name).strip()
                if not pw_id:
                    continue
                cls_name = f"Pathway_{pw_id}"
                pw_class = self.onto[cls_name]
                if not pw_class:
                    with self.onto:
                        pw_class = type(cls_name, (self.BiologicalPathway,), {})
                mol_instance.is_involved_in_pathway.append(pw_class())

            # GO term annotations (optional)
            go_terms = features.get('go_terms', []) or []
            for term in go_terms:
                term_id = str(term).strip()
                if not term_id:
                    continue
                term_name = term_id.replace(':', '_')
                go_class = self.onto[term_name]
                if not go_class:
                    with self.onto:
                        go_class = type(term_name, (self.GOTerm,), {})
                go_instance = go_class()
                mol_instance.hasGOTerm.append(go_instance)

            # MeSH term annotations (optional)
            mesh_terms = features.get('mesh_terms', []) or []
            for term in mesh_terms:
                term_id = str(term).strip()
                if not term_id:
                    continue
                term_name = term_id.replace(':', '_')
                mesh_class = self.onto[term_name]
                if not mesh_class:
                    with self.onto:
                        mesh_class = type(term_name, (self.MeshTerm,), {})
                mesh_instance = mesh_class()
                mol_instance.hasMeshTerm.append(mesh_instance)

            # Generic annotation terms (dataset/ontology context)
            annotation_terms = features.get('annotation_terms', []) or []
            for term in annotation_terms:
                term_id = str(term).strip()
                if not term_id:
                    continue
                term_name = term_id.replace(':', '_')
                anno_class = self.onto[term_name]
                if not anno_class:
                    with self.onto:
                        anno_class = type(term_name, (self.AnnotationTerm,), {})
                anno_instance = anno_class()
                mol_instance.hasAnnotationTerm.append(anno_instance)
                    
            # Aromatic Rings
            if features['has_aromatic']:
                mol_instance.is_a.append(self.AromaticMolecule)
            else:
                mol_instance.is_a.append(self.NonAromaticMolecule)

            # Label
            mol_instance.hasLabel = [label]
            
            return mol_instance
    
    def save(self):
        """온톨로지를 파일로 저장"""
        self.onto.save(file=self.ontology_path, format="rdfxml")
        print(f"Ontology saved to {self.ontology_path}")
    
    def load(self):
        """온톨로지를 파일에서 로드"""
        if os.path.exists(self.ontology_path):
            self.onto = get_ontology(self.ontology_path).load()
            print(f"Ontology loaded from {self.ontology_path}")
        else:
            print(f"Ontology file not found: {self.ontology_path}")
