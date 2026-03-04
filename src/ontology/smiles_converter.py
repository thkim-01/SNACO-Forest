"""
SMILES Converter: SMILES 문자열을 분자 특성으로 변환하는 모듈
"""
import json
import os
import sqlite3
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.SaltRemover import SaltRemover
from typing import Dict, List, Optional

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except Exception:
    rdMolStandardize = None


_FEATURE_VERSION = 4  # Bump when feature schema changes to invalidate cache


class MolecularFeatureExtractor:
    """SMILES로부터 의미적 분자 특성을 추출하는 클래스"""
    
    def __init__(self, cache_path: Optional[str] = None):
        """Create a feature extractor.

        Args:
            cache_path: Optional path to a SQLite cache DB. When provided,
                extracted features will be persisted and reused across runs.
        """

        self.cache_path = cache_path
        self._mem_cache: Dict[str, Dict] = {}
        self._db_conn: Optional[sqlite3.Connection] = None
        self._pending_writes: int = 0
        self._commit_every: int = 50

        if self.cache_path:
            self._init_cache_db(self.cache_path)

    def _init_cache_db(self, cache_path: str) -> None:
        Path(os.path.dirname(cache_path) or ".").mkdir(
            parents=True, exist_ok=True
        )
        conn = sqlite3.connect(cache_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                smiles TEXT PRIMARY KEY,
                features_json TEXT,
                error TEXT
            )
            """
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.commit()
        self._db_conn = conn

    @staticmethod
    def _normalize_smiles_key(smiles: str) -> str:
        """Normalize SMILES key for caching (strip only).

        Full standardization (salt removal + neutralization + canonicalization)
        is done in `smiles_to_mol` where a valid RDKit Mol is available.
        """
        return str(smiles).strip()

    @staticmethod
    def _neutralize_atoms(mol: Chem.Mol) -> Chem.Mol:
        """Neutralize charged atoms where chemically reasonable."""
        pattern = Chem.MolFromSmarts(
            "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
        )
        if mol is None or pattern is None:
            return mol
        matches = mol.GetSubstructMatches(pattern)
        if not matches:
            return mol

        rw = Chem.RWMol(mol)
        for match in matches:
            atom = rw.GetAtomWithIdx(match[0])
            charge = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(max(0, int(hcount - charge)))
            atom.UpdatePropertyCache()
        return rw.GetMol()

    def _cache_get(self, key: str) -> Optional[Dict]:
        if key in self._mem_cache:
            hit = self._mem_cache[key]
            if hit.get('_version') == _FEATURE_VERSION:
                return hit
            # stale version → re-extract
            return None

        if not self._db_conn:
            return None

        cur = self._db_conn.execute(
            "SELECT features_json, error FROM features WHERE smiles = ?",
            (key,),
        )
        row = cur.fetchone()
        if not row:
            return None

        features_json, error = row
        if error:
            raise ValueError(error)

        if not features_json:
            return None

        features = json.loads(features_json)
        # Invalidate stale cache entries when feature schema changes
        if features.get('_version') != _FEATURE_VERSION:
            return None
        self._mem_cache[key] = features
        return features

    def _cache_set(
        self,
        key: str,
        features: Optional[Dict] = None,
        error: Optional[str] = None,
    ) -> None:
        if features is not None:
            self._mem_cache[key] = features

        if not self._db_conn:
            return

        features_json = (
            json.dumps(features, ensure_ascii=False)
            if features is not None
            else None
        )
        self._db_conn.execute(
            "INSERT OR REPLACE INTO features(smiles, features_json, error) "
            "VALUES (?, ?, ?)",
            (key, features_json, error),
        )
        self._pending_writes += 1
        if self._pending_writes >= self._commit_every:
            self._db_conn.commit()
            self._pending_writes = 0

    def close(self) -> None:
        """Flush pending cache writes and close DB connection."""
        if not self._db_conn:
            return
        try:
            self._db_conn.commit()
        finally:
            self._db_conn.close()
            self._db_conn = None
            self._pending_writes = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Avoid destructor-time exceptions.
            pass
    
    def smiles_to_mol(self, smiles: str) -> Chem.Mol:
        """SMILES 문자열을 RDKit Mol 객체로 변환"""
        smiles = str(smiles).strip()

        # 1) Parse
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # 2) Explicit salt / solvent removal (Na+, K+, Ca2+, nitrate, acetate, water, ethanol...)
        remover = SaltRemover()
        stripped = remover.StripMol(mol, dontRemoveEverything=True)
        if stripped is not None and stripped.GetNumAtoms() > 0:
            mol = stripped

        # 3) Neutralization for descriptor consistency
        mol = self._neutralize_atoms(mol)

        # 4) Canonicalization to stable parent representation
        if rdMolStandardize is not None:
            try:
                uncharger = rdMolStandardize.Uncharger()
                mol = uncharger.uncharge(mol)
            except Exception:
                pass

        canonical = Chem.MolToSmiles(mol, canonical=True)
        mol = Chem.MolFromSmiles(canonical)
        if mol is None:
            raise ValueError(f"Failed canonicalization for SMILES: {smiles}")
        return mol
    
    def extract_features(self, smiles: str) -> Dict:
        """SMILES로부터 모든 의미적 특성을 추출"""
        key = self._normalize_smiles_key(smiles)
        if self.cache_path:
            cached = self._cache_get(key)
            if cached is not None:
                return cached

        try:
            mol = self.smiles_to_mol(key)
        except Exception as e:
            if self.cache_path:
                self._cache_set(key, error=str(e))
            raise
        
        features = {
            '_version': _FEATURE_VERSION,
            # 기본 속성
            'molecular_weight': Descriptors.MolWt(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_heavy_atoms': Lipinski.HeavyAtomCount(mol),
            'num_rotatable_bonds': Lipinski.NumRotatableBonds(mol),
            
            # H-bond 특성
            'num_hba': Lipinski.NumHAcceptors(mol),
            'num_hbd': Lipinski.NumHDonors(mol),
            
            # 링 구조
            'num_rings': Lipinski.RingCount(mol),
            'num_aromatic_rings': Lipinski.NumAromaticRings(mol),
            'has_aromatic': Lipinski.NumAromaticRings(mol) > 0,
            
            # LogP (친유성)
            'logp': Descriptors.MolLogP(mol),
            
            # TPSA (Topological Polar Surface Area)
            'tpsa': Descriptors.TPSA(mol),
            
            # Lipinski Rule of Five
            'obeys_lipinski': self._check_lipinski(mol),
            
            # Functional Groups (기본 + 구조 알럿)
            'functional_groups': self._detect_functional_groups(mol),
            
            # ---- v2 확장 속성 ----
            # 몰 굴절률 (Molar Refractivity)
            'molar_refractivity': Crippen.MolMR(mol),
            
            # 질소+산소 원자 합계 (N+O count)
            'n_plus_o_count': Lipinski.NOCount(mol),
            
            # 이종 원자 수
            'num_heteroatoms': Lipinski.NumHeteroatoms(mol),
            
            # 형식 전하 (Formal Charge)
            'formal_charge': Chem.GetFormalCharge(mol),
            
            # sp3 탄소 분율 (Fraction CSP3)
            'fsp3': rdMolDescriptors.CalcFractionCSP3(mol),
            
            # 구조 알럿 (Structural Alerts) - 독성/약리 관련
            'structural_alerts': self._detect_structural_alerts(mol),
        }

        # BACE core linkage candidates for ontology object-property population
        binding_core_candidates = {
            'Amidine',
            'Guanidine',
            'Acylguanidine',
            'Iminohydantoin',
            'Hydroxyethylamine',
        }
        alerts = set(features.get('structural_alerts', []) or [])
        fgroups = set(features.get('functional_groups', []) or [])
        features['binding_cores'] = sorted(list((alerts | fgroups) & binding_core_candidates))
        
        # 카테고리화
        features['mw_category'] = self._categorize_molecular_weight(
            features['molecular_weight']
        )
        features['logp_category'] = self._categorize_logp(features['logp'])
        features['tpsa_category'] = self._categorize_tpsa(features['tpsa'])

        if self.cache_path:
            self._cache_set(key, features=features)

        return features
    
    def _detect_functional_groups(self, mol: Chem.Mol) -> List[str]:
        """분자 내 functional group 검출 (SMARTS 패턴 기반)"""
        detected = []
        
        # SMARTS 패턴 정의
        patterns = {
            'Amine': '[NX3;H2,H1;!$(NC=O)]',
            'Alcohol': '[OX2H]',
            'Carbonyl': '[CX3]=[OX1]',
            'Carboxyl': '[CX3](=O)[OX2H1]',
            'Ether': '[OD2]([#6])[#6]',
            'Ester': '[#6][CX3](=O)[OX2H0][#6]',
            'Amide': '[NX3][CX3](=[OX1])[#6]',
            'Halogen': '[F,Cl,Br,I]',
            'Aromatic': 'c',
            'Sulfur': '[#16]',
            'Nitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
            # v2 확장: 약리/독성 관련 작용기
            'Phenol': '[OX2H]c1ccccc1',
            'Epoxide': 'C1OC1',
            'MichaelAcceptor': '[#6]=[#6][CX3]=[OX1]',
            'AromaticAmine': '[NX3;H2,H1;!$(NC=O)]c',
            'AromaticNitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])]c',
            'Quinone': '[#6]1(=[OX1])[#6]=[#6][#6](=[OX1])[#6]=[#6]1',
            'Thiophene': 'c1ccsc1',
            'Imidazole': 'c1cnc[nH]1',
            'Triazole': 'c1nnn[nH]1',
            'Amidine': '[NX3][CX3]=[NX2]',
            'Guanidine': '[NX3][CX3](=[NX2])[NX3]',
            'Acylguanidine': '[NX3][CX3](=[NX2])[NX3][CX3]=[OX1]',
            'Hydroxyethylamine': '[NX3]CC[OX2H]',
        }
        
        for group_name, smarts in patterns.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                detected.append(group_name)
        
        return detected

    # ---- v2: 구조 알럿 (Structural Alerts) ----
    # SMARTS-기반 약리/독성 구조 알럿 패턴

    _STRUCTURAL_ALERT_PATTERNS = {
        # --- 독성 알럿 (ClinTox / Tox21) ---
        'MichaelAcceptor': '[#6]=[#6][CX3]=[OX1]',             # α,β-불포화 카보닐
        'Epoxide': 'C1OC1',                                     # 에폭사이드 (3원자 고리)
        'AromaticAmine': '[NX3;H2,H1;!$(NC=O)]c',              # 방향족 아민
        'AromaticNitro': '[$([NX3](=O)=O),$([NX3+](=O)[O-])]c', # 니트로 방향족
        'Quinone': '[#6]1(=[OX1])[#6]=[#6][#6](=[OX1])[#6]=[#6]1', # 퀴논
        'Thiophene': 'c1ccsc1',                                  # 티오펜 고리
        'Alkylhalide': '[CX4][F,Cl,Br,I]',                      # 알킬할로겐화물 (Cramer III)
        'Acylhalide': '[CX3](=[OX1])[F,Cl,Br,I]',              # 아실할로겐화물
        'StrongElectrophile_Mustard': 'ClCCN',                   # 질소 머스타드 (DNA 알킬화)
        'Hydrazine': '[NX3][NX3]',                               # 하이드라진

        # --- BACE 관련 (효소 저해제 골격) ---
        'Amidine': '[NX3][CX3]=[NX2]',                          # 아미딘 코어
        'Guanidine': '[NX3][CX3](=[NX2])[NX3]',                 # 구아니딘 코어
        'Acylguanidine': '[NX3][CX3](=[NX2])[NX3][CX3]=[OX1]',  # 아실구아니딘
        'Hydroxyethylamine': '[NX3]CC[OX2H]',                    # 하이드록시에틸아민
        'Iminohydantoin': 'O=C1NC(=N)NC1=O',                    # 이미노하이단토인

        # --- BBB / 수송체 관련 ---
        'Phenol': '[OX2H]c1ccccc1',                              # 페놀
        'QuaternaryN': '[NX4+]',                                 # 4급 암모늄 (영구전하)
        'Imidazole': 'c1cnc[nH]1',                               # 이미다졸
        'Triazole': 'c1nnn[nH]1',                                # 트리아졸

        # --- HIV 관련 ---
        'Nucleoside': 'OC1CCCO1',                                # 리보스/디옥시리보스 유사
        'MetalChelator': '[#8,#7,#16]~[#6]~[#8,#7,#16]',        # 금속 킬레이트 코어 (IN 저해제)

        # --- SIDER / 부작용 관련 ---
        'Carboxylicacid': '[CX3](=O)[OX2H1]',                   # 카복실산 (NSAID/GI 독성)
        'Sulfonamide': '[SX4](=[OX1])(=[OX1])[NX3]',            # 설폰아마이드
        'Steroid': 'C1CCC2C(C1)CCC1C2CCC2CCCCC12',              # 스테로이드 4-고리 골격 근사

        # --- BBBP / 수송체 기질 관련 (v4) ---
        'AlphaAminoAcid': '[NX3;H2,H1][CX4H]C(=O)[OX2H1,OX1-]', # α-아미노산 스캐폴드 (LAT1 mimicry)
        'LipophilicCation': '[CX4,c]~[CX4,c]~[CX4,c]~[*+1]',   # 친유성 양이온 (SR-MMP)

        # --- Tox21 / 경로 관련 (v4) ---
        'PlanarPAH': 'c1ccc2c(c1)ccc1ccccc12',                   # 평면형 다환 방향족 (NR-AhR)

        # --- BACE / 선택성 관련 (v4) ---
        'PeptideLike': '[NX3][CX4H][CX3](=[OX1])[NX3][CX4H][CX3](=[OX1])',  # 디펩티드 유사 구조 (CatD 교차반응)
    }

    def _detect_structural_alerts(self, mol: Chem.Mol) -> List[str]:
        """구조 알럿(Structural Alert) 패턴 검출.

        Returns:
            검출된 알럿 이름 리스트
        """
        detected: List[str] = []
        for alert_name, smarts in self._STRUCTURAL_ALERT_PATTERNS.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    detected.append(alert_name)
            except Exception:
                continue
        return detected
    
    def _check_lipinski(self, mol: Chem.Mol) -> bool:
        """Lipinski's Rule of Five 체크"""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        
        return (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
    
    def _categorize_molecular_weight(self, mw: float) -> str:
        """분자량을 카테고리로 분류"""
        if mw < 200:
            return 'Low'
        elif mw < 400:
            return 'Medium'
        else:
            return 'High'
    
    def _categorize_logp(self, logp: float) -> str:
        """LogP를 카테고리로 분류 (친수성/친유성)"""
        if logp < 0:
            return 'Hydrophilic'
        elif logp < 3:
            return 'Moderate'
        else:
            return 'Lipophilic'
    
    def _categorize_tpsa(self, tpsa: float) -> str:
        """TPSA를 카테고리로 분류"""
        if tpsa < 60:
            return 'Low'
        elif tpsa < 140:
            return 'Medium'
        else:
            return 'High'


class MolecularInstance:
    """개별 분자 인스턴스를 나타내는 클래스"""
    
    def __init__(self, mol_id: str, smiles: str, label: int, features: Dict):
        self.mol_id = mol_id
        self.smiles = smiles
        self.label = label
        self.features = features
    
    def __repr__(self):
        return f"MolecularInstance(id={self.mol_id}, label={self.label})"
    
    def satisfies_refinement(self, refinement) -> bool:
        """
        이 분자가 특정 refinement를 만족하는지 확인
        refinement는 (property, operator, value) 형태
        """
        prop, operator, value = refinement
        
        if prop not in self.features:
            return False
        
        feature_value = self.features[prop]
        
        if operator == '==':
            return feature_value == value
        elif operator == '>':
            return feature_value > value
        elif operator == '>=':
            return feature_value >= value
        elif operator == '<':
            return feature_value < value
        elif operator == '<=':
            return feature_value <= value
        elif operator == 'contains':
            # functional_groups와 같은 리스트 속성
            if isinstance(feature_value, list):
                return value in feature_value
            return False
        
        return False
