# flake8: noqa

"""Multi-dataset evaluation for Semantic Bagging Forest.

This script evaluates the bagging ensemble on multiple MoleculeNet-style
classification datasets included in `data/`.

By default it runs one representative task per dataset to keep runtime practical.
Use `--all-tasks` to evaluate all tasks for multi-task datasets (Tox21, SIDER).
"""

import argparse
import csv
import os
import re
import sys
import time
import traceback
from pathlib import Path

# MUST BE FIRST: Initialize paths for all imports
from _init_paths import init_paths
if not init_paths():
    raise SystemExit(1)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Optional, List, Dict
from tabulate import tabulate

# Now import src modules (paths initialized above)
from src.ontology.molecule_ontology import MoleculeOntology
from src.ontology.smiles_converter import MolecularFeatureExtractor
from src.ontology.concept_extractor import OntologyConceptExtractor
from src.sdt.logic_forest import SemanticForest

FIXED_RANDOM_STATE = 42

# Tox21 assay -> gene mapping (core targets)
TOX21_TASK_GENE_MAP = {
    "NR-AR": "AR",
    "NR-AR-LBD": "AR",
    "NR-AhR": "AHR",
    "NR-Aromatase": "CYP19A1",
    "NR-ER": "ESR1",
    "NR-ER-LBD": "ESR1",
    "NR-PPAR-gamma": "PPARG",
    "SR-ARE": "NFE2L2",
    "SR-ATAD5": "ATAD5",
    "SR-HSE": "HSF1",
    "SR-p53": "TP53",
    "SR-MMP": "MMP",
}

# Tox21 target gene -> GO terms (core subset)
TOX21_GENE_GO_MAP = {
    "AR": ["GO:0003677", "GO:0004871", "GO:0005634", "GO:0043401"],
    "ESR1": ["GO:0003677", "GO:0004871", "GO:0005634", "GO:0030522"],
    "PPARG": ["GO:0003677", "GO:0004871", "GO:0005634", "GO:0042981"],
    "AHR": ["GO:0003700", "GO:0005634", "GO:0010807", "GO:0006805"],
    "CYP19A1": ["GO:0004497", "GO:0005506", "GO:0006691", "GO:0008202"],
    "NFE2L2": ["GO:0003700", "GO:0006950", "GO:0042535", "GO:0006979"],
    "ATAD5": ["GO:0003677", "GO:0006281", "GO:0006974", "GO:0005515"],
    "HSF1": ["GO:0003700", "GO:0006950", "GO:0009408", "GO:0034605"],
    "TP53": ["GO:0003677", "GO:0006915", "GO:0000077", "GO:0006281"],
    "MMP": ["GO:0043538", "GO:0005739", "GO:0007005", "GO:0042773"],
}

# LBD tasks: add generic binding activity term
TOX21_LBD_TASKS = {"NR-AR-LBD", "NR-ER-LBD"}
GO_BINDING_ACTIVITY = "GO:0005488"

# BACE target gene -> GO terms (core subset)
BACE_GENE_GO_MAP = {
    "BACE1": [
        "GO:0004190",  # aspartic-type endopeptidase activity
        "GO:0042803",  # protein homodimerization activity
        "GO:0042987",  # amyloid precursor protein catabolic process
        "GO:0006508",  # proteolysis
        "GO:0010001",  # glial cell differentiation
        "GO:0005768",  # endosome
        "GO:0005802",  # trans-Golgi network
    ]
}

# ClinTox task -> GO/MeSH context (core subset)
CLINTOX_TASK_GO_MAP = {
    "CT_TOX": [
        "GO:0006950",  # response to stress
        "GO:0006979",  # response to oxidative stress
        "GO:0006915",  # apoptotic process
        "GO:0006281",  # DNA repair
    ],
    "FDA_APPROVED": [
        "GO:0008150",  # biological_process (generic baseline)
        "GO:0004888",  # transmembrane signaling receptor activity
    ],
}

CLINTOX_TASK_MESH_MAP = {
    "CT_TOX": [
        "MESH_TOXICITY",
        "MESH_DRUG_TOXICITY",
    ],
    "FDA_APPROVED": [
        "MESH_FDA_APPROVAL",
        "MESH_DRUG_APPROVAL",
    ],
}

# HIV task -> GO/MeSH context (core subset)
HIV_TASK_GO_MAP = {
    "HIV_active": [
        "GO:0006955",  # immune response
        "GO:0016032",  # viral process
        "GO:0045087",  # innate immune response
        "GO:0009615",  # response to virus
    ]
}

HIV_TASK_MESH_MAP = {
    "HIV_active": [
        "MESH_HIV_INFECTIONS",
        "MESH_ANTIVIRAL_AGENTS",
    ]
}


def _get_tox21_go_terms(task_name: str) -> List[str]:
    gene = TOX21_TASK_GENE_MAP.get(task_name)
    if not gene:
        return []
    terms = list(TOX21_GENE_GO_MAP.get(gene, []))
    if task_name in TOX21_LBD_TASKS:
        terms.append(GO_BINDING_ACTIVITY)
    return sorted(set(terms))


def _get_bace_go_terms(task_name: str) -> List[str]:
    # Single-task dataset: apply BACE1 context to all samples.
    if str(task_name).lower() != "class":
        return []
    return sorted(set(BACE_GENE_GO_MAP.get("BACE1", [])))


def _get_clintox_go_terms(task_name: str) -> List[str]:
    return sorted(set(CLINTOX_TASK_GO_MAP.get(task_name, [])))


def _get_hiv_go_terms(task_name: str) -> List[str]:
    return sorted(set(HIV_TASK_GO_MAP.get(task_name, [])))


def _get_clintox_mesh_terms(task_name: str) -> List[str]:
    return sorted(set(CLINTOX_TASK_MESH_MAP.get(task_name, [])))


def _get_hiv_mesh_terms(task_name: str) -> List[str]:
    return sorted(set(HIV_TASK_MESH_MAP.get(task_name, [])))


def _get_go_terms_for_task(dataset_key: str, task_name: str) -> List[str]:
    if str(dataset_key).lower() == "tox21":
        return _get_tox21_go_terms(task_name)
    if str(dataset_key).lower() == "bace":
        return _get_bace_go_terms(task_name)
    if str(dataset_key).lower() == "clintox":
        return _get_clintox_go_terms(task_name)
    if str(dataset_key).lower() == "hiv":
        return _get_hiv_go_terms(task_name)
    return []


def _get_mesh_terms_for_task(dataset_key: str, task_name: str) -> List[str]:
    if str(dataset_key).lower() == "clintox":
        return _get_clintox_mesh_terms(task_name)
    if str(dataset_key).lower() == "sider":
        return _get_sider_mesh_terms(task_name)
    if str(dataset_key).lower() == "hiv":
        return _get_hiv_mesh_terms(task_name)
    return []


def _get_sider_mesh_terms(task_name: str) -> List[str]:
    # Default: map each SIDER task to a synthetic MeSH-like term for context.
    safe = _safe_name(task_name)
    if not safe:
        return []
    return [f"MESH_SIDER_{safe}"]


DATASET_ONTOLOGY_CONTEXT = {
    "tox21": [
        "ONTO_BAO",
        "ONTO_DTO",
        "ONTO_GO",
        "ONTO_CHEBI",
        "ONTO_PATO",
    ],
    "bace": [
        "ONTO_DTO",
        "ONTO_CHEBI",
    ],
    "clintox": [
        "ONTO_MESH",
        "ONTO_BAO",
        "ONTO_GO",
        "ONTO_CHEBI",
        "ONTO_THESAURUS",
    ],
    "hiv": [
        "ONTO_BAO",
        "ONTO_GO",
        "ONTO_CHEBI",
    ],
    "sider": [
        "ONTO_MESH",
        "ONTO_THESAURUS",
        "ONTO_PATO",
        "ONTO_CHEBI",
    ],
    "bbbp": [
        "ONTO_CHEBI",
    ],
}


def _get_annotation_terms_for_dataset(dataset_key: str, dl_config: Optional[Dict] = None) -> List[str]:
    cfg = dl_config or {}
    context_map = cfg.get("dataset_ontology_context", {})
    values = context_map.get(str(dataset_key).lower(), DATASET_ONTOLOGY_CONTEXT.get(str(dataset_key).lower(), []))

    normalized = []
    for v in values:
        name = str(v).strip().upper()
        if not name:
            continue
        if not name.startswith("ONTO_"):
            name = f"ONTO_{name}"
        normalized.append(name)
    return normalized


def _load_dl_config(config_path: str) -> Dict:
    """Load DL/Reasoner configuration YAML with safe fallbacks."""
    defaults = {
        "reasoning": {
            "engine": "none",
            "infer_property_values": False,
            "enable_toxicity_seed_rules": False,
        },
        "ontology_paths": {
            "BAO": "ontology/bao_complete.owl",
            "DTO": "ontology/DTO.owl",
            "GO": "ontology/go.owl",
            "CHEBI": "ontology/chebi.owl",
            "PATO": "ontology/pato.owl",
            "MESH": "ontology/mesh.owl",
            "THESAURUS": "ontology/Thesaurus.owl",
        },
        "dataset_primary_ontology": {
            "bbbp": "ontology/chebi.owl",
            "bace": "ontology/DTO.owl",
            "clintox": "ontology/DTO.owl",
            "hiv": "ontology/bao_complete.owl",
            "tox21": "ontology/bao_complete.owl",
            "sider": "ontology/mesh.owl",
        },
        "dataset_ontology_context": {
            "tox21": ["BAO", "DTO", "GO", "CHEBI", "PATO"],
            "bace": ["DTO", "CHEBI"],
            "clintox": ["MESH", "BAO", "GO", "CHEBI", "THESAURUS"],
            "hiv": ["BAO", "GO", "CHEBI"],
            "sider": ["MESH", "THESAURUS", "PATO", "CHEBI"],
            "bbbp": ["CHEBI"],
        },
        "dataset_refinement_profile": {
            "bbbp": {
                "allowed_ref_types": ["concept", "cardinality", "domain", "qualification"],
                "allowed_object_properties": ["hasFunctionalGroupRel", "hasRingSystem"],
                "allowed_data_properties": ["hasMolecularWeight", "hasLogP", "hasTPSA", "hasNumHBA", "hasNumHBD", "hasNumRotatableBonds", "hasNumRings", "hasNumAromaticRings", "hasAromaticity"],
                "max_qualification_concepts_per_property": 32,
            },
            "tox21": {
                "allowed_ref_types": ["concept", "cardinality", "domain", "qualification", "conjunction"],
                "max_qualification_concepts_per_property": 24,
            },
            "clintox": {
                "allowed_ref_types": ["concept", "cardinality", "domain", "qualification", "conjunction"],
                "max_qualification_concepts_per_property": 24,
            },
            "hiv": {
                "allowed_ref_types": ["concept", "cardinality", "domain", "qualification"],
                "max_qualification_concepts_per_property": 24,
            },
            "sider": {
                "allowed_ref_types": ["concept", "cardinality", "domain", "qualification"],
                "max_qualification_concepts_per_property": 20,
            },
            "bace": {
                "allowed_ref_types": ["concept", "cardinality", "domain", "qualification", "conjunction"],
                "max_qualification_concepts_per_property": 24,
            },
        },
    }

    p = Path(config_path)
    if not p.exists():
        return defaults

    loaded: Dict = {}
    suffix = str(p.suffix).lower()
    if suffix not in (".yaml", ".yml"):
        print("[WARN] Only YAML is supported for --dl-config. Using defaults.")
        return defaults

    try:
        import yaml  # type: ignore
    except Exception:
        print("[WARN] PyYAML is not installed. Using defaults.")
        return defaults

    try:
        parsed = yaml.safe_load(p.read_text(encoding="utf-8"))
        loaded = parsed if isinstance(parsed, dict) else {}
    except Exception:
        return defaults

    merged = dict(defaults)
    for key in ("reasoning", "ontology_paths", "dataset_primary_ontology", "dataset_ontology_context", "dataset_refinement_profile"):
        merged[key] = dict(defaults.get(key, {}))
        merged[key].update(loaded.get(key, {}))
    return merged


def _get_ontology_paths_for_dataset(dataset_key: str, dl_config: Optional[Dict] = None) -> Dict[str, str]:
    """Build ontology path mapping required by a dataset.

    Returns a dict like {'GO': 'ontology/go.owl', 'DTO': 'ontology/DTO.owl', ...}
    limited to ontologies configured in DATASET_ONTOLOGY_CONTEXT.
    """
    cfg = dl_config or {}
    all_paths = dict(cfg.get("ontology_paths", {}))
    if not all_paths:
        all_paths = {
            "BAO": "ontology/bao_complete.owl",
            "DTO": "ontology/DTO.owl",
            "GO": "ontology/go.owl",
            "CHEBI": "ontology/chebi.owl",
            "PATO": "ontology/pato.owl",
            "MESH": "ontology/mesh.owl",
            "THESAURUS": "ontology/Thesaurus.owl",
        }

    selected = {}
    dataset_context = cfg.get("dataset_ontology_context", {})
    ctx_values = dataset_context.get(str(dataset_key).lower(), _get_annotation_terms_for_dataset(dataset_key))
    for raw_name in ctx_values:
        name = str(raw_name).replace("ONTO_", "").upper()
        if name in all_paths and Path(all_paths[name]).exists():
            # Normalize key capitalization used by concept extractor
            norm_name = "Thesaurus" if name == "THESAURUS" else name
            selected[norm_name] = all_paths[name]
    return selected


def _safe_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name))
    return s[:80].strip("_") or "task"


# NOTE: commit_sha / commit_message tracking was intentionally removed.
# The benchmark outputs are now purely metrics-focused.


def _normalize_binary_labels(series: pd.Series) -> pd.Series:
    """Normalize common binary label encodings.

    - drops missing values
    - treats -1 as missing (common in some datasets)
    - supports {-1, 1} by mapping to {0, 1} after dropping missing
    """
    y = pd.to_numeric(series, errors="coerce")
    # In some MoleculeNet dumps, -1 is used as missing.
    y = y.replace(-1, np.nan)
    y = y.dropna()

    uniq = set(y.unique().tolist())
    if uniq == {1.0, 0.0}:
        return y.astype(int)
    if uniq == {1.0, -1.0}:
        # If -1 survived (shouldn't), map it.
        return y.replace(-1, 0).astype(int)
    if uniq == {1.0} or uniq == {0.0}:
        return y.astype(int)

    # Best-effort: if values are already 0/1-like
    if uniq.issubset({0.0, 1.0}):
        return y.astype(int)

    raise ValueError(f"Non-binary labels found: {sorted(uniq)}")


def populate_ontology(
    onto: MoleculeOntology,
    extractor: MolecularFeatureExtractor,
    df: pd.DataFrame,
    smiles_col: str,
    label_col: str,
    subset_name: str,
    go_terms: Optional[List[str]] = None,
    mesh_terms: Optional[List[str]] = None,
    annotation_terms: Optional[List[str]] = None,
    concept_extractor: Optional[OntologyConceptExtractor] = None,
    dataset_key: Optional[str] = None,
    dataset_concepts: Optional[Dict[str, List]] = None,
):
    instances = []
    labels = []

    def _augment_relation_features(feats: Dict, ds_key: Optional[str], task_name: str, label_val: int) -> Dict:
        """Attach dataset-specific relation features for ontology object properties.

        Implements the full dataset-specific ontology mapping guideline (v4):
        - bbbp: is_substrate_of (P-gp, LAT1), has_physicochemical_risk
        - bace: hasBindingCore, interactsWithResidue, selectivity filter
        - clintox: isMemberOfEDTClass, hasReactiveToxicophore, Cramer class
        - hiv: inhibitsStep, targetsProtein (PR, RT, IN)
        - tox21: hasAOP, is_involved_in_pathway (MMP/AhR)
        - sider: LLT→PT→SOC hierarchy + mapsToPrimarySOC (multi-axiality)
        """
        ds = str(ds_key or "").lower()
        out = dict(feats)
        alerts = set(out.get("structural_alerts", []) or [])
        fgroups = set(out.get("functional_groups", []) or [])

        # ==================================================================
        # 1. BBBP – 수송체 기질 및 물리화학적 리스크
        # ==================================================================
        if ds == "bbbp":
            n_plus_o = out.get("n_plus_o_count", 0)
            mw = out.get("molecular_weight", 0)

            substrates: List[str] = []
            # P-gp substrate rule: (N+O) >= 8, MW > 400
            if n_plus_o >= 8 and mw > 400:
                substrates.append("PgpSubstrate")
            # LAT1 mimicry: α-amino acid scaffold SMARTS hit
            if "AlphaAminoAcid" in alerts:
                substrates.append("LAT1_Substrate_Mimic")
            out["transporter_substrates"] = substrates

            # Physicochemical penetration risks
            risks: List[str] = []
            if mw > 500:
                risks.append("HighMW_Risk")
            if out.get("tpsa", 0) > 120:
                risks.append("HighTPSA_Risk")
            if n_plus_o >= 10:
                risks.append("HighNO_Risk")
            out["physicochemical_risks"] = risks

        # ==================================================================
        # 2. BACE – 결합 코어 + 잔기 상호작용 + 선택성 필터
        # ==================================================================
        elif ds == "bace":
            out["binding_cores"] = list(out.get("binding_cores", []) or [])

            # Catalytic dyad / residue interactions
            residues: List[str] = []
            if {"Amidine", "Acylguanidine"} & alerts:
                residues.extend(["Asp32", "Asp228"])
            if "Guanidine" in alerts:
                residues.append("Gly230")
            if "Hydroxyethylamine" in alerts:
                residues.extend(["Asp32", "Asp228"])
            out["interacts_residues"] = sorted(set(residues))

            # Selectivity filter: large peptide-like → CatD cross-reaction
            if "PeptideLike" in alerts and out.get("molecular_weight", 0) > 600:
                out["selectivity_class"] = "Low_Selectivity_Inhibitor"
            elif out.get("binding_cores"):
                out["selectivity_class"] = "AsparticProtease_Active"
            else:
                out["selectivity_class"] = ""

        # ==================================================================
        # 3. ClinTox – FDA EDT / Cramer 클래스 + 반응성 독성단
        # ==================================================================
        elif ds == "clintox":
            # EDT class (I=safest … VI=most hazardous)
            edt = "EDT_Class_I"
            if {"MichaelAcceptor", "StrongElectrophile_Mustard"} & alerts:
                edt = "EDT_Class_VI"
            elif {"Epoxide", "Quinone"} & alerts:
                edt = "EDT_Class_V"
            elif "Hydrazine" in alerts:
                edt = "EDT_Class_IV"
            elif {"AromaticAmine", "AromaticNitro"} & alerts:
                edt = "EDT_Class_III"
            out["edt_class"] = edt

            # Cramer class (I=low, II=moderate, III=high)
            cramer = "Cramer_Class_I"
            if {"Alkylhalide", "Acylhalide", "MichaelAcceptor", "Epoxide", "Quinone"} & alerts:
                cramer = "Cramer_Class_III"
            elif "AromaticAmine" in alerts:
                cramer = "Cramer_Class_II"
            out["cramer_class"] = cramer

            # Reactive toxicophores
            toxicophores = [t for t in [
                "MichaelAcceptor", "Epoxide", "Quinone",
                "Hydrazine", "StrongElectrophile_Mustard",
            ] if t in alerts]
            out["reactive_toxicophores"] = toxicophores

        # ==================================================================
        # 4. HIV – 바이러스 복제 단계 + 타겟 단백질 (N:M)
        # ==================================================================
        elif ds == "hiv":
            steps: List[str] = []
            proteins: List[str] = []

            # RT (Reverse Transcriptase)
            if "Nucleoside" in alerts:
                steps.append("Reverse_Transcription")
                proteins.append("RT")
            # IN (Integrase)
            if "MetalChelator" in alerts:
                steps.append("Integration")
                proteins.append("IN")
            # PR (Protease) – Asp25 interaction patterns
            pr_motifs = {"Hydroxyethylamine", "Acylguanidine", "Guanidine", "Amidine"}
            if (pr_motifs & alerts) or ("Amide" in fgroups):
                steps.append("Maturity")
                proteins.append("PR")
            # Assembly heuristic
            if "Halogen" in fgroups and out.get("logp", 0.0) > 2.0:
                steps.append("Assembly")
            # Default fallback
            if not steps:
                steps.append("Replication")
                proteins.append("RT")

            out["inhibits_steps"] = sorted(set(steps))
            out["targets_proteins"] = sorted(set(proteins))

        # ==================================================================
        # 5. Tox21 – AOP 경로 + NR/SR 생물학적 경로 + MMP/AhR
        # ==================================================================
        elif ds == "tox21":
            assay_to_aop = {
                "NR-AR": ["AOP_AR_Modulation"],
                "NR-AR-LBD": ["AOP_AR_LBD_Binding"],
                "NR-AhR": ["AOP_AhR_Activation"],
                "NR-Aromatase": ["AOP_Aromatase_Inhibition"],
                "NR-ER": ["AOP_ER_Modulation"],
                "NR-ER-LBD": ["AOP_ER_LBD_Binding"],
                "NR-PPAR-gamma": ["AOP_PPARg_Modulation"],
                "SR-ARE": ["AOP_Oxidative_Stress_Nrf2"],
                "SR-ATAD5": ["AOP_DNA_Damage_Replication_Stress"],
                "SR-HSE": ["AOP_HeatShock_Protein_Stress"],
                "SR-MMP": ["AOP_Mitochondrial_Dysfunction"],
                "SR-p53": ["AOP_p53_DNA_Damage_Response"],
            }
            aops = assay_to_aop.get(str(task_name), [])
            is_active = int(label_val) == 1
            out["aop_terms"] = list(aops) if is_active else []

            # Biological pathway enrichment
            pathways: List[str] = []
            # SR-MMP: lipophilic cation → MMP_Disruptor
            if "LipophilicCation" in alerts or (
                out.get("logp", 0) > 3 and out.get("formal_charge", 0) > 0
            ):
                pathways.append("MMP_Disruptor")
            # NR-AhR: planar PAH → AhR_Ligand
            if "PlanarPAH" in alerts:
                pathways.append("AhR_Ligand")
            # General NR / SR pathway categorisation
            task_str = str(task_name)
            if task_str.startswith("NR-"):
                pathways.append("NR_Pathway")
            elif task_str.startswith("SR-"):
                pathways.append("SR_Pathway")
            out["pathway_terms"] = pathways if is_active else []

        # ==================================================================
        # 6. SIDER – LLT→PT→SOC 계층 + Primary SOC 우선순위
        # ==================================================================
        elif ds == "sider":
            # MedDRA Primary SOC priority (higher index = higher priority)
            _SOC_PRIORITY = {
                "Congenital_familial_and_genetic_disorders": 27,
                "Pregnancy_puerperium_and_perinatal_conditions": 26,
                "Surgical_and_medical_procedures": 25,
                "Social_circumstances": 24,
                "Infections_and_infestations": 23,
                "Neoplasms_benign_malignant_and_unspecified": 22,
                "Blood_and_lymphatic_system_disorders": 21,
                "Immune_system_disorders": 20,
                "Endocrine_disorders": 19,
                "Metabolism_and_nutrition_disorders": 18,
                "Psychiatric_disorders": 17,
                "Nervous_system_disorders": 16,
                "Eye_disorders": 15,
                "Ear_and_labyrinth_disorders": 14,
                "Cardiac_disorders": 13,
                "Vascular_disorders": 12,
                "Respiratory_thoracic_and_mediastinal_disorders": 11,
                "Gastrointestinal_disorders": 10,
                "Hepatobiliary_disorders": 9,
                "Skin_and_subcutaneous_tissue_disorders": 8,
                "Musculoskeletal_and_connective_tissue_disorders": 7,
                "Renal_and_urinary_disorders": 6,
                "Reproductive_system_and_breast_disorders": 5,
                "General_disorders_and_administration_site_conditions": 4,
                "Investigations": 3,
                "Injury_poisoning_and_procedural_complications": 2,
                "Product_issues": 1,
            }

            if int(label_val) == 1:
                safe_task = _safe_name(task_name)
                soc_name = f"SOC_{safe_task}"
                pt_name = f"PT_{safe_task}"
                llt_name = f"LLT_{safe_task}"

                out["sider_soc_terms"] = [soc_name]
                out["sider_pt_terms"] = [pt_name]
                out["sider_llt_terms"] = [llt_name]

                # Multi-axiality: determine primary SOC via priority lookup
                best_soc, best_pri = soc_name, 0
                for soc_candidate, pri in _SOC_PRIORITY.items():
                    if soc_candidate.lower() in safe_task.lower() or safe_task.lower() in soc_candidate.lower():
                        if pri > best_pri:
                            best_soc, best_pri = f"SOC_{soc_candidate}", pri
                out["primary_soc"] = best_soc
            else:
                out["sider_soc_terms"] = []
                out["sider_pt_terms"] = []
                out["sider_llt_terms"] = []
                out["primary_soc"] = ""

        return out

    for idx, row in df.iterrows():
        try:
            smi = row[smiles_col]
            feats = extractor.extract_features(smi)
            label_val = int(row[label_col])

            feats = _augment_relation_features(
                feats=feats,
                ds_key=dataset_key,
                task_name=label_col,
                label_val=label_val,
            )
            if go_terms:
                feats = dict(feats)
                feats["go_terms"] = list(go_terms)
            if mesh_terms:
                feats = dict(feats)
                feats["mesh_terms"] = list(mesh_terms)
            if annotation_terms:
                feats = dict(feats)
                feats["annotation_terms"] = list(annotation_terms)

            # Per-molecule ontology concept attachments for dataset-specific DL branching
            if concept_extractor and dataset_key and dataset_concepts:
                feats = dict(feats)
                attachments = concept_extractor.get_molecule_conceptattachments(
                    molecule_features=feats,
                    dataset_name=str(dataset_key).lower(),
                    dataset_concepts=dataset_concepts,
                )

                existing_ann = list(feats.get("annotation_terms", []) or [])
                for onto_name, concept_names in attachments.items():
                    # Prefix to avoid collisions and preserve ontology provenance
                    existing_ann.extend([f"{onto_name}_{str(c)}" for c in concept_names])
                feats["annotation_terms"] = existing_ann

            mol_id = f"Mol_{subset_name}_{idx}"
            inst = onto.add_molecule_instance(mol_id, feats, label=label_val)
            instances.append(inst)
            labels.append(label_val)
        except Exception:
            # Skip invalid SMILES / feature extraction failures.
            continue

    return instances, labels


def _get_dataset_ontology_path(dataset_key: str, dl_config: Optional[Dict] = None) -> Optional[str]:
    """Select appropriate ontology file for each dataset.
    
    Mapping:
    - BBBP → chebi.owl (화학 구조)
    - BACE → DTO.owl (약물-단백질 타깃)
    - ClinTox → DTO.owl, chebi.owl (약물, 화학)
    - HIV → bao_complete.owl (바이러스, 생물학적 경로)
    - Tox21 → bao_complete.owl, pato.owl, go.owl (독성, 생물학적 경로)
    - SIDER → mesh.owl (임상 부작용)
    """
    cfg = dl_config or {}
    ontology_map = cfg.get("dataset_primary_ontology", {})
    if not ontology_map:
        ontology_map = {
            "bbbp": "ontology/chebi.owl",
            "bace": "ontology/DTO.owl",
            "clintox": "ontology/DTO.owl",  # Could also use chebi.owl
            "hiv": "ontology/bao_complete.owl",       # Could also use NCIT
            "tox21": "ontology/bao_complete.owl",     # Could also use pato.owl or go.owl
            "sider": "ontology/mesh.owl",
        }
    
    path = ontology_map.get(dataset_key.lower())
    if path and Path(path).exists():
        return path
    return None  # Fall back to default DTO.xrdf


def _resolve_tree_param(cli_value, dataset_key: str, param_name: str,
                         fallback: int, dl_config: Optional[Dict] = None) -> int:
    """Resolve a tree hyper-parameter: CLI override → config dataset_tree_params → fallback.

    This allows per-dataset optimal depth/split/leaf settings from the YAML config
    while still permitting CLI overrides for quick experimentation.
    """
    if cli_value is not None:
        return int(cli_value)
    cfg = (dl_config or {}).get("dataset_tree_params", {})
    ds_params = cfg.get(str(dataset_key).lower(), {})
    return int(ds_params.get(param_name, fallback))


def evaluate_task(
    dataset_key: str,
    dataset_name: str,
    csv_path: str,
    smiles_col: str,
    label_col: str,
    feature_cache_path: Optional[str],
    split_criterion: str,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    sample_size: int,
    test_size: float,
    random_state: int,
    compute_backend: str,
    torch_device: str,
    heuristic_probe_samples: int,
    dl_config: Optional[Dict] = None,
):
    df = pd.read_csv(csv_path)

    if smiles_col not in df.columns:
        raise ValueError(f"Missing smiles column '{smiles_col}' in {csv_path}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {csv_path}")

    # Normalize labels (drop missing)
    y_norm = _normalize_binary_labels(df[label_col])
    df = df.loc[y_norm.index].copy()
    df[label_col] = y_norm

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col] if df[label_col].nunique() > 1 else None,
    )

    onto_path = Path("ontology") / f"temp_bagging_{dataset_key}_{_safe_name(label_col)}.owl"
    if onto_path.exists():
        onto_path.unlink()

    # Get dataset-specific ontology
    base_onto_path = _get_dataset_ontology_path(dataset_key, dl_config=dl_config)
    
    onto = MoleculeOntology(str(onto_path), base_dto_path=base_onto_path)
    extractor = MolecularFeatureExtractor(cache_path=feature_cache_path)

    # Dataset-specific ontology concept extraction for DL refinement prioritization
    ontology_paths = _get_ontology_paths_for_dataset(dataset_key, dl_config=dl_config)
    concept_extractor = OntologyConceptExtractor(ontology_paths) if ontology_paths else None
    dataset_concepts = {}
    if concept_extractor:
        dataset_concepts = concept_extractor.get_dataset_relevant_concepts(
            dataset_name=str(dataset_key).lower(),
            ontology_names=list(ontology_paths.keys()),
        )

    go_terms = _get_go_terms_for_task(dataset_key, label_col)
    mesh_terms = _get_mesh_terms_for_task(dataset_key, label_col)
    annotation_terms = _get_annotation_terms_for_dataset(dataset_key, dl_config=dl_config)

    reasoning_cfg = (dl_config or {}).get("reasoning", {})
    refinement_profile_cfg = (dl_config or {}).get("dataset_refinement_profile", {})
    reasoner_engine = str(reasoning_cfg.get("engine", "none"))
    infer_property_values = bool(reasoning_cfg.get("infer_property_values", False))
    enable_toxicity_seed_rules = bool(reasoning_cfg.get("enable_toxicity_seed_rules", False))
    dataset_refinement_profile = dict(refinement_profile_cfg.get(str(dataset_key).lower(), {}))

    try:
        train_instances, _ = populate_ontology(
            onto,
            extractor,
            train_df,
            smiles_col,
            label_col,
            subset_name="Train",
            go_terms=go_terms,
            mesh_terms=mesh_terms,
            annotation_terms=annotation_terms,
            concept_extractor=concept_extractor,
            dataset_key=dataset_key,
            dataset_concepts=dataset_concepts,
        )
        test_instances, test_labels = populate_ontology(
            onto,
            extractor,
            test_df,
            smiles_col,
            label_col,
            subset_name="Test",
            go_terms=go_terms,
            mesh_terms=mesh_terms,
            annotation_terms=annotation_terms,
            concept_extractor=concept_extractor,
            dataset_key=dataset_key,
            dataset_concepts=dataset_concepts,
        )

        # If feature extraction filtered too much, bail out.
        if len(train_instances) < max(min_samples_split, 50) or len(test_instances) < 50:
            return {
                "dataset": dataset_name,
                "task": label_col,
                "n_train": len(train_instances),
                "n_test": len(test_instances),
                "auc": np.nan,
                "acc": np.nan,
                "note": "too_few_valid_instances",
            }

        forest = SemanticForest(
            onto,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            verbose=False,
            dataset_name=str(dataset_key).lower(),
            dataset_concepts=dataset_concepts,
            learner_kwargs={
                "split_criterion": split_criterion,
                "search_strategy": "exhaustive",
                "compute_backend": compute_backend,
                "torch_device": torch_device,
                "random_state": random_state,
                "heuristic_probe_samples": heuristic_probe_samples,
                "enable_toxicity_seed_rules": enable_toxicity_seed_rules,
                "reasoner_engine": reasoner_engine,
                "reasoner_infer_property_values": infer_property_values,
                "dataset_refinement_profile": dataset_refinement_profile,
            },
        )
        forest.fit(train_instances)

        probs = forest.predict_proba(test_instances)
        preds = forest.predict(test_instances)

        acc = accuracy_score(test_labels, preds)
        try:
            auc = roc_auc_score(test_labels, probs)
        except ValueError:
            auc = 0.5

        return {
            "dataset": dataset_name,
            "task": label_col,
            "n_train": len(train_instances),
            "n_test": len(test_instances),
            "auc": float(auc),
            "acc": float(acc),
            "note": "",
        }
    finally:
        extractor.close()


def _append_result(out_path: Path, res: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "task",
        "n_train",
        "n_test",
        "auc",
        "acc",
        "note",
    ]
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: res.get(k, "") for k in fieldnames})


def _write_dataset_averages(in_path: Path, out_path: Path) -> None:
    """Write macro averages per dataset over completed tasks.

    - Excludes rows with NaN metrics from mean/std computations.
    - Keeps counts so partial/resumed runs are still meaningful.
    """
    if not in_path.exists() or in_path.stat().st_size == 0:
        return

    try:
        df = pd.read_csv(in_path)
    except Exception:
        return

    required = {
        "dataset",
        "task",
        "n_train",
        "n_test",
        "auc",
        "acc",
        "note",
    }
    if not required.issubset(set(df.columns)):
        return

    # Coerce numeric columns
    for c in ["n_train", "n_test", "auc", "acc"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Only compute stats on rows with finite metrics.
    valid = df[df["auc"].notna() & df["acc"].notna()].copy()

    rows = []

    for dataset, df_ds in df.groupby("dataset"):
        df_valid = valid[valid["dataset"] == dataset]

        rows.append(
            {
                "dataset": dataset,
                "n_tasks": int(df_ds["task"].nunique()),
                "n_rows": int(len(df_ds)),
                "n_valid": int(len(df_valid)),
                "auc_mean": float(df_valid["auc"].mean()) if len(df_valid) else np.nan,
                "auc_std": float(df_valid["auc"].std(ddof=0)) if len(df_valid) else np.nan,
                "acc_mean": float(df_valid["acc"].mean()) if len(df_valid) else np.nan,
                "acc_std": float(df_valid["acc"].std(ddof=0)) if len(df_valid) else np.nan,
                "n_train_mean": float(df_valid["n_train"].mean()) if len(df_valid) else np.nan,
                "n_test_mean": float(df_valid["n_test"].mean()) if len(df_valid) else np.nan,
            }
        )

    # Overall macro average across all valid rows.
    if len(valid):
        rows.append(
            {
                "dataset": "ALL",
                "n_tasks": int(df["task"].nunique()),
                "n_rows": int(len(df)),
                "n_valid": int(len(valid)),
                "auc_mean": float(valid["auc"].mean()),
                "auc_std": float(valid["auc"].std(ddof=0)),
                "acc_mean": float(valid["acc"].mean()),
                "acc_std": float(valid["acc"].std(ddof=0)),
                "n_train_mean": float(valid["n_train"].mean()),
                "n_test_mean": float(valid["n_test"].mean()),
            }
        )

    if not rows:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["dataset"]).reset_index(drop=True)
    out_df.to_csv(out_path, index=False)


def _acquire_benchmark_lock(lock_path: Path) -> None:
    """Acquire a single-run lock to prevent concurrent benchmarks.

    We've observed accidental double-starts (e.g., venv python + system python)
    which can corrupt output CSVs. This lock makes the benchmark single-instance.

    If a stale lock is found (PID not running), it is removed automatically.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()

    def _pid_running(p: int) -> bool:
        if p <= 0:
            return False
        try:
            # On Windows, os.kill(pid, 0) works to check existence.
            os.kill(p, 0)
            return True
        except Exception:
            return False

    # Try exclusive create.
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid={pid}\n")
            f.write(f"started_at={time.time()}\n")
        return
    except FileExistsError:
        # Possible concurrent run or stale lock.
        try:
            txt = lock_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            txt = ""
        other_pid = 0
        m = re.search(r"pid=(\d+)", txt)
        if m:
            try:
                other_pid = int(m.group(1))
            except Exception:
                other_pid = 0

        if other_pid and _pid_running(other_pid):
            raise SystemExit(
                f"Benchmark already running (lock={lock_path}, pid={other_pid})."
            )

        # Stale lock: remove and retry once.
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid={pid}\n")
            f.write(f"started_at={time.time()}\n")
        return


def _release_benchmark_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Semantic Bagging Forest across datasets"
    )
    parser.add_argument("--n-estimators", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Max tree depth. If omitted, uses dataset-specific default from config.")
    parser.add_argument("--min-samples-split", type=int, default=None,
                        help="Min samples to split. If omitted, uses dataset-specific default from config.")
    parser.add_argument("--min-samples-leaf", type=int, default=None,
                        help="Min samples per leaf. If omitted, uses dataset-specific default from config.")
    parser.add_argument("--sample-size", type=int, default=0, help="Max samples per dataset (0=no limit, use all data)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--random-state",
        type=int,
        default=FIXED_RANDOM_STATE,
        help="Deprecated: ignored. Random seed is fixed to 42 for reproducibility.",
    )
    parser.add_argument(
        "--split-criterion",
        default="information_gain",
        choices=["information_gain", "id3", "gain_ratio", "c45_gain_ratio", "gini"],
        help=(
            "Split criterion for tree growth. 'information_gain' (default) and 'id3' use "
            "ID3-style information gain; 'gain_ratio' matches C4.5's gain ratio; "
            "'gini' uses CART's Gini impurity."
        ),
    )
    parser.add_argument(
        "--algorithm",
        default=None,
        choices=["id3", "c45", "cart"],
        help=(
            "Algorithm alias for split criterion. "
            "id3 -> information_gain, c45 -> gain_ratio, cart -> gini. "
            "If provided, this overrides --split-criterion."
        ),
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Evaluate all tasks for multi-task datasets (Tox21, SIDER).",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "Comma-separated dataset keys to run (e.g., 'clintox' or 'tox21,sider'). "
            "If omitted, runs all datasets."
        ),
    )
    parser.add_argument(
        "--out",
        default=str(Path("output") / "semantic_forest_benchmark.csv"),
        help="Output CSV path.",
    )

    parser.add_argument(
        "--out-avg",
        default=None,
        help=(
            "Optional output CSV path for per-dataset macro averages over tasks. "
            "If omitted, uses '<out>_avg.csv'."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output CSV if it already exists (disables resume).",
    )
    parser.add_argument(
        "--feature-cache-dir",
        default=str(Path("output") / "feature_cache"),
        help=(
            "Directory for persistent SMILES->features cache (SQLite). "
            "Set to empty string to disable caching."
        ),
    )
    parser.add_argument(
        "--compute-backend",
        default="auto",
        choices=["auto", "numpy", "torch"],
        help="Compute backend for impurity calculations.",
    )
    parser.add_argument(
        "--search-strategy",
        default="exhaustive",
        choices=["exhaustive"],
        help="Refinement candidate search strategy per node.",
    )
    parser.add_argument("--heuristic-probe-samples", type=int, default=128)
    parser.add_argument(
        "--torch-device",
        default="auto",
        help="Torch device when --compute-backend torch/auto (e.g., auto, cpu, cuda).",
    )
    parser.add_argument(
        "--dl-config",
        default=str(Path("configs") / "dl_reasoner_config.yaml"),
        help="Path to DL/Reasoner YAML config (dataset-ontology mapping + reasoner engine).",
    )
    args = parser.parse_args()
    args.random_state = FIXED_RANDOM_STATE

    dl_config = _load_dl_config(args.dl_config)

    if args.algorithm:
        algo_to_criterion = {
            "id3": "information_gain",
            "c45": "gain_ratio",
            "cart": "gini",
        }
        args.split_criterion = algo_to_criterion[args.algorithm]

    out_path = Path(args.out)
    out_avg_path = (
        Path(args.out_avg)
        if args.out_avg
        else out_path.with_name(out_path.stem + "_avg.csv")
    )

    # Single-run lock to avoid concurrent benchmarks corrupting CSV outputs.
    lock_path = out_path.with_suffix(out_path.suffix + ".lock")
    _acquire_benchmark_lock(lock_path)
    if args.overwrite and out_path.exists():
        out_path.unlink()

    completed: set[tuple[str, str]] = set()
    if out_path.exists() and not args.overwrite:
        try:
            prev = pd.read_csv(out_path)
            if {"dataset", "task"}.issubset(set(prev.columns)):
                # Resume: only skip tasks already present in the current output file.
                completed = set(zip(prev["dataset"], prev["task"]))
        except Exception:
            completed = set()

    datasets = [
        {
            "key": "bbbp",
            "name": "BBBP",
            "path": "data/bbbp/BBBP.csv",
            "smiles": "smiles",
            "tasks": ["p_np"],
        },
        {
            "key": "bace",
            "name": "BACE",
            "path": "data/bace/bace.csv",
            "smiles": "smiles",
            "tasks": ["Class"],
        },
        {
            "key": "clintox",
            "name": "ClinTox",
            "path": "data/clintox/clintox.csv",
            "smiles": "smiles",
            "tasks": ["CT_TOX", "FDA_APPROVED"],
            "default_task": "CT_TOX",
        },
        {
            "key": "hiv",
            "name": "HIV",
            "path": "data/hiv/HIV.csv",
            "smiles": "smiles",
            "tasks": ["HIV_active"],
        },
        {
            "key": "tox21",
            "name": "Tox21",
            "path": "data/tox21/tox21.csv",
            "smiles": "smiles",
            "tasks": [
                "NR-AR",
                "NR-AR-LBD",
                "NR-AhR",
                "NR-Aromatase",
                "NR-ER",
                "NR-ER-LBD",
                "NR-PPAR-gamma",
                "SR-ARE",
                "SR-ATAD5",
                "SR-HSE",
                "SR-MMP",
                "SR-p53",
            ],
            "default_task": "SR-p53",
        },
        {
            "key": "sider",
            "name": "SIDER",
            "path": "data/sider/sider.csv",
            "smiles": "smiles",
            "tasks": None,  # determined from header
            "default_task": "Hepatobiliary disorders",
        },
    ]

    if args.datasets:
        available_keys = [d["key"] for d in datasets]
        wanted = {
            s.strip().lower()
            for s in str(args.datasets).split(",")
            if s.strip()
        }
        datasets = [ds for ds in datasets if ds["key"].lower() in wanted]
        if not datasets:
            raise ValueError(
                f"No datasets matched --datasets={args.datasets!r}. "
                f"Available: {available_keys}"
            )

    # Collect results for real-time table display
    results_buffer = []
    
    def _print_results_table():
        """Print current results as a formatted table."""
        if not results_buffer:
            return
        df = pd.DataFrame(results_buffer)
        # Format floats for readability
        for col in ['auc', 'acc']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A")
        print("\n" + "="*120)
        print("BENCHMARK PROGRESS TABLE")
        print("="*120)
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
        print("="*120 + "\n")
    
    try:
        for ds in datasets:
            tasks = ds.get("tasks")
            if tasks is None:
                # SIDER: all columns except smiles
                df_cols = pd.read_csv(ds["path"], nrows=1).columns.tolist()
                tasks = [c for c in df_cols if c != ds["smiles"]]

            if not args.all_tasks:
                # Use one representative task for multi-task datasets.
                if "default_task" in ds:
                    tasks = [ds["default_task"]]

            for task in tasks:
                if (ds["name"], task) in completed:
                    print(f"Skipping (already in output): {ds['name']}/{task}")
                    continue

                try:
                    cache_dir = str(args.feature_cache_dir).strip()
                    cache_path = None
                    if cache_dir:
                        cache_path = str(Path(cache_dir) / f"{ds['key']}.sqlite3")

                    res = evaluate_task(
                        dataset_key=ds["key"],
                        dataset_name=ds["name"],
                        csv_path=ds["path"],
                        smiles_col=ds["smiles"],
                        label_col=task,
                        feature_cache_path=cache_path,
                        split_criterion=args.split_criterion,
                        n_estimators=args.n_estimators,
                        max_depth=_resolve_tree_param(
                            args.max_depth, ds["key"], "max_depth", 10, dl_config),
                        min_samples_split=_resolve_tree_param(
                            args.min_samples_split, ds["key"], "min_samples_split", 20, dl_config),
                        min_samples_leaf=_resolve_tree_param(
                            args.min_samples_leaf, ds["key"], "min_samples_leaf", 5, dl_config),
                        sample_size=args.sample_size,
                        test_size=args.test_size,
                        random_state=args.random_state,
                        compute_backend=args.compute_backend,
                        torch_device=args.torch_device,
                        heuristic_probe_samples=args.heuristic_probe_samples,
                        dl_config=dl_config,
                    )
                except Exception as e:
                    print(f"[ERROR] {ds['name']}/{task} failed: {e}")
                    traceback.print_exc()
                    res = {
                        "dataset": ds["name"],
                        "task": task,
                        "n_train": 0,
                        "n_test": 0,
                        "auc": np.nan,
                        "acc": np.nan,
                        "note": f"error: {e}",
                    }

                _append_result(out_path, res)
                completed.add((ds["name"], task))
                
                # Add to results buffer and print table
                results_buffer.append(res)
                _print_results_table()
                
                # Also log to stdout for posterity
                print(
                    f"[Completed] {res['dataset']}/{res['task']}: "
                    f"AUC={res['auc']:.4f}, ACC={res['acc']:.4f} "
                    f"(n_train={res['n_train']}, n_test={res['n_test']}) {res['note']}"
                )
    finally:
        # Always refresh the macro-average file, even on interruption.
        _write_dataset_averages(out_path, out_avg_path)
        print(f"\nSaved: {out_path}")
        print(f"Saved averages: {out_avg_path}")

        _release_benchmark_lock(lock_path)


if __name__ == "__main__":
    main()
