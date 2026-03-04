# -*- coding: utf-8 -*-
"""
Dataset-specific ontology-driven seed rules for Semantic Decision Trees.

각 데이터셋의 도메인 지식(물리화학적 임계값, 구조 알럿, 약전계 매핑)을
OntologyRefinement 객체로 인코딩하여, 트리의 초기 분할 후보에 주입한다.

참조:
- BBBP: 혈뇌장벽 투과성 (TPSA, LogP, MW, HBD 기반)
- BACE: BACE-1 효소 저해제 (아미딘/구아니딘 코어, 포켓 적합성)
- ClinTox: 임상 독성 (FDA EDT, 반응성 작용기 알럿)
- HIV: HIV 복제 저해 (메커니즘별 온톨로지, 하이브리드 지문)
- Tox21: 12종 멀티태스크 독성 (NR/SR 분석별 알럿)
- SIDER: 시판 약물 부작용 (MedDRA SOC 해부학적 분류)
"""

from typing import List, Dict, Any, Optional

from src.sdt.logic_refinement import OntologyRefinement


# ---------------------------------------------------------------------------
# Helper: 간결하게 OntologyRefinement 생성
# ---------------------------------------------------------------------------

def _domain(prop: str, op: str, val: Any) -> OntologyRefinement:
    """Data property threshold rule (domain refinement)."""
    return OntologyRefinement('domain', property_name=prop, operator=op, value=val)


def _concept(onto, class_name: str) -> Optional[OntologyRefinement]:
    """Class membership rule (concept refinement)."""
    cls = getattr(onto, class_name, None)
    if cls is None:
        return None
    return OntologyRefinement('concept', operator='is_a', concept=cls)


def _qual(onto, prop_name: str, class_name: str) -> Optional[OntologyRefinement]:
    """Existential qualification rule: ∃ prop.Class."""
    cls = getattr(onto, class_name, None)
    if cls is None:
        return None
    return OntologyRefinement('qualification', property_name=prop_name, concept=cls)


def _fg(onto, group_name: str) -> Optional[OntologyRefinement]:
    """Shortcut for hasFunctionalGroupRel qualification."""
    return _qual(onto, 'hasFunctionalGroupRel', group_name)


def _collect(*rules: Optional[OntologyRefinement]) -> List[OntologyRefinement]:
    """None을 제거한 유효한 규칙만 수집."""
    return [r for r in rules if r is not None]


# ============================================================================
# BBBP: Blood-Brain Barrier Penetration
# 핵심 전략: 물리화학적 임계값 필터링 > 수송체 기질 알럿
# ============================================================================

def get_bbbp_rules(onto) -> List[OntologyRefinement]:
    """BBBP 15개 초기 시드 규칙.

    BBB 투과성은 주로 수동 확산에 의해 결정되며,
    TPSA < 90 Å², 1 < LogP < 5, MW < 450 Da가 핵심 임계값이다.
    """
    return _collect(
        # 1. TPSA 투과성 규칙: TPSA < 60 이면 BBB+ 확률 85%↑
        _domain('hasTPSA', '<', 60.0),
        # 2. TPSA 중간 임계값: TPSA < 90 (일반적 BBB 투과 상한)
        _domain('hasTPSA', '<', 90.0),
        # 3. 지질 친화성 최적구간 하한: LogP > 1
        _domain('hasLogP', '>', 1.0),
        # 4. 지질 친화성 최적구간 상한: LogP < 4
        _domain('hasLogP', '<', 4.0),
        # 5. 과다 친유성 제한: LogP > 5 이면 비특이적 결합 위험
        _domain('hasLogP', '>', 5.0),
        # 6. 분자량 제한: MW > 500 이면 BBB- 강제 분할
        _domain('hasMolecularWeight', '>', 500.0),
        # 7. 분자량 완화 임계값: MW > 450 (투과 효율 급감 시작)
        _domain('hasMolecularWeight', '>', 450.0),
        # 8. 수소 결합 공여자 제한: HBD > 3 이면 투과성 저하
        _domain('hasNumHBD', '>', 3),
        # 9. 수소 결합 수용자 제한: HBA > 10 (Lipinski 위반)
        _domain('hasNumHBA', '>', 10),
        # 10. 유연성 제한: Rotatable Bonds > 8 이면 엔트로피 손실
        _domain('hasNumRotatableBonds', '>', 8),
        # 11. N+O 원자 합계: N+O >= 5 이면 BBB+ 확률 감소
        _domain('hasNPlusOCount', '>', 5),
        # 12. 몰 굴절률 상한: MR > 130 이면 BBB-
        _domain('hasMolarRefractivity', '>', 130.0),
        # 13. 몰 굴절률 하한: MR < 40 이면 BBB-
        _domain('hasMolarRefractivity', '<', 40.0),
        # 14. 영구 전하 규칙: 4급 암모늄 포함 시 BBB-
        _fg(onto, 'QuaternaryN'),
        # 15. 방향족 분류: AromaticMolecule 여부
        _concept(onto, 'AromaticMolecule'),
    )


# ============================================================================
# BACE: Beta-Secretase 1 저해제
# 핵심 전략: 핵심 약전계(Pharmacophore) 매핑 > 선택성 필터
# ============================================================================

def get_bace_rules(onto) -> List[OntologyRefinement]:
    """BACE 15개 초기 시드 규칙.

    BACE-1의 촉매 부위(Asp32/Asp228)와의 상호작용이 핵심이며,
    아미딘/구아니딘 코어, 소수성 포켓 점유, 선택성 필터가 주요 분할 기준.
    """
    return _collect(
        # 1. 촉매 부위 결합: 아미딘 코어 포함 시 활성 분할
        _fg(onto, 'Amidine'),
        # 2. 촉매 부위 결합: 구아니딘 코어 포함 시 활성 분할
        _fg(onto, 'Guanidine'),
        # 3. 아실구아니딘 코어: 고효율 리간드
        _fg(onto, 'Acylguanidine'),
        # 4. 펩티드 모방체: 하이드록시에틸아민 코어 활성 분할
        _fg(onto, 'Hydroxyethylamine'),
        # 5. 이미노하이단토인 코어: 고효율 리간드
        _fg(onto, 'Iminohydantoin'),
        # 6. 아마이드 결합 = Gly230 수소 결합 패턴
        _fg(onto, 'Amide'),
        # 7. 사이드 체인 불소 치환: 할로겐 존재 시 활성+BBB 고려
        _fg(onto, 'Halogen'),
        # 8. 방향족 고리 개수: AromaticRings > 2 (S1/S3 포켓 점유)
        _domain('hasNumAromaticRings', '>', 2),
        # 9. 고리형 구조 복잡도: NumRings > 3 (큰 결합 포켓 적합)
        _domain('hasNumRings', '>', 3),
        # 10. 분자 크기 하한: MW > 300 (최소 부피 확보)
        _domain('hasMolecularWeight', '>', 300.0),
        # 11. 분자 크기 상한: MW > 600 (과대 분자 비활성 경향)
        _domain('hasMolecularWeight', '>', 600.0),
        # 12. 소수성: LogP > 2 (소수성 포켓 결합력)
        _domain('hasLogP', '>', 2.0),
        # 13. TPSA > 100 (용해도 vs 투과성 트레이드오프)
        _domain('hasTPSA', '>', 100.0),
        # 14. HBD > 2 (수소 결합 패턴 분석)
        _domain('hasNumHBD', '>', 2),
        # 15. 아민 존재: 염기성 아민 (BACE 결합 일반 패턴)
        _fg(onto, 'Amine'),
    )


# ============================================================================
# ClinTox: Clinical Toxicity
# 핵심 전략: FDA EDT 규제 클래스 > 반응성 작용기 알럿
# ============================================================================

def get_clintox_rules(onto) -> List[OntologyRefinement]:
    """ClinTox 15개 초기 시드 규칙.

    임상 실패의 강력한 지표인 반응성 작용기(Michael Acceptor, Epoxide)와
    FDA EDT 분류 기준(Cramer Class III 등)을 반영.
    """
    return _collect(
        # 1. Michael Acceptor: α,β-불포화 카보닐 → 독성 분할
        _fg(onto, 'MichaelAcceptor'),
        # 2. 에폭사이드: 고반응성 3원자 고리 → 세포 독성
        _fg(onto, 'Epoxide'),
        # 3. 방향족 아민: 유전독성 알럿
        _fg(onto, 'AromaticAmine'),
        # 4. 방향족 니트로: 유전독성 알럿
        _fg(onto, 'AromaticNitro'),
        # 5. 퀴논: 산화환원 사이클링 독성
        _fg(onto, 'Quinone'),
        # 6. 니트로기: 일반 독성 알럿
        _fg(onto, 'Nitro'),
        # 7. 티오펜 고리: bioactivation으로 반응성 대사체 형성
        _fg(onto, 'Thiophene'),
        # 8. 알킬할로겐화물: Cramer Class III 고반응성 구조
        _fg(onto, 'Alkylhalide'),
        # 9. 하이드라진: 유전독성 의심 구조
        _fg(onto, 'Hydrazine'),
        # 10. 할로겐 일반: 독성 기여 가능 작용기
        _fg(onto, 'Halogen'),
        # 11. 분자량 과대: MW > 500 이면 독성 발현 위험 증가
        _domain('hasMolecularWeight', '>', 500.0),
        # 12. 과다 친유성: LogP > 5 이면 비특이적 독성 위험
        _domain('hasLogP', '>', 5.0),
        # 13. 과다 유연성: RotBonds > 10 → 대사 불안정
        _domain('hasNumRotatableBonds', '>', 10),
        # 14. TPSA > 140: 높은 극성 → 독성 프로파일 변화
        _domain('hasTPSA', '>', 140.0),
        # 15. 고리 개수 과다: NumRings > 4 → 복잡도 관련 독성
        _domain('hasNumRings', '>', 4),
    )


# ============================================================================
# HIV: HIV 복제 저해제
# 핵심 전략: 복제 주기 메커니즘 분류 > 하이브리드 지문 분할
# ============================================================================

def get_hiv_rules(onto) -> List[OntologyRefinement]:
    """HIV 15개 초기 시드 규칙.

    RT(역전사효소), PR(프로테아제), IN(인테그레이즈) 등 다양한 타겟을 반영.
    대규모 데이터셋(40K+)이므로 넓은 화학 공간 커버리지가 핵심.
    """
    return _collect(
        # 1. 방향족 분류: 대부분의 활성 화합물은 방향족 구조 보유
        _concept(onto, 'AromaticMolecule'),
        # 2. 이미다졸 고리: 다양한 항바이러스제 핵심 골격
        _fg(onto, 'Imidazole'),
        # 3. 아마이드 결합: 펩티드 모방체/PR 저해제
        _fg(onto, 'Amide'),
        # 4. 아민: 염기성 아민 (PR/IN 저해제 공통 특성)
        _fg(onto, 'Amine'),
        # 5. 할로겐: 선택성 향상 치환기
        _fg(onto, 'Halogen'),
        # 6. 금속 킬레이트 코어: IN 가닥 전이 저해제 패턴
        _fg(onto, 'MetalChelator'),
        # 7. 뉴클레오사이드 유사: NRTI 패턴
        _fg(onto, 'Nucleoside'),
        # 8. 분자량 중간: MW > 300 (최소 활성 분자 크기)
        _domain('hasMolecularWeight', '>', 300.0),
        # 9. 분자량 상한: MW > 500 (대형 분자 활성 패턴)
        _domain('hasMolecularWeight', '>', 500.0),
        # 10. 소수성: LogP > 2 (RT NNRTI 소수성 포켓)
        _domain('hasLogP', '>', 2.0),
        # 11. 과다 친유성: LogP > 4 (비특이적 결합 위험)
        _domain('hasLogP', '>', 4.0),
        # 12. 방향족 고리: AromaticRings > 2 (활성 화합물 경향)
        _domain('hasNumAromaticRings', '>', 2),
        # 13. HBA > 5 (수소 결합 수용자 풍부 → 결합력)
        _domain('hasNumHBA', '>', 5),
        # 14. TPSA > 90 (극성 표면적과 타겟 결합)
        _domain('hasTPSA', '>', 90.0),
        # 15. 고리 개수: NumRings > 3 (다고리 활성 화합물)
        _domain('hasNumRings', '>', 3),
    )


# ============================================================================
# Tox21: Toxicology in the 21st Century (12종 멀티태스크)
# 핵심 전략: 공통 화학 공간 분할 > 분석별 독성 알럿
# ============================================================================

def get_tox21_rules(onto) -> List[OntologyRefinement]:
    """Tox21 15개 초기 시드 규칙.

    NR 7종(AhR, AR, AR-LBD, Aromatase, ER, ER-LBD, PPAR-γ) +
    SR 5종(ARE, ATAD5, HSE, MMP, p53) 멀티태스크.
    클래스 불균형(IR > 28)이 극심하므로 구조적 알럿에 의존.
    """
    return _collect(
        # 1. NR-AhR: 평면형 다환 방향족 → AromaticMolecule
        _concept(onto, 'AromaticMolecule'),
        # 2. 니트로기: NR-AhR / SR-ARE 독성 알럿
        _fg(onto, 'Nitro'),
        # 3. 방향족 아민: 유전독성 / 내분비 교란
        _fg(onto, 'AromaticAmine'),
        # 4. Michael Acceptor: SR-ARE (항산화 반응) 활성
        _fg(onto, 'MichaelAcceptor'),
        # 5. 퀴논: SR-ARE / SR-MMP 산화환원 사이클링
        _fg(onto, 'Quinone'),
        # 6. 페놀: NR-ER (에스트로겐 수용체) 에스트라디올 모방
        _fg(onto, 'Phenol'),
        # 7. 이미다졸: NR-Aromatase 저해제 패턴
        _fg(onto, 'Imidazole'),
        # 8. 트리아졸: NR-Aromatase 저해제 패턴
        _fg(onto, 'Triazole'),
        # 9. 방향족 니트로: 니트로아로마틱 유전독성
        _fg(onto, 'AromaticNitro'),
        # 10. 할로겐: 내분비 교란 물질 공통 작용기
        _fg(onto, 'Halogen'),
        # 11. 친유성: LogP > 3 → SR-MMP 활성 (미토콘드리아 독성)
        _domain('hasLogP', '>', 3.0),
        # 12. 극성 표면적: TPSA > 80 (일반 독성 관련)
        _domain('hasTPSA', '>', 80.0),
        # 13. 분자량: MW > 400 (고분자량 독성 경향)
        _domain('hasMolecularWeight', '>', 400.0),
        # 14. 고리 개수: NumRings > 3 (복잡 구조 독성)
        _domain('hasNumRings', '>', 3),
        # 15. HBD > 2 (수소 결합 공여자 패턴)
        _domain('hasNumHBD', '>', 2),
    )


# ============================================================================
# SIDER: Side Effect Resource (27개 SOC 부작용)
# 핵심 전략: 해부학적 계통 분류 > 약물-타겟 상호작용 매핑
# ============================================================================

def get_sider_rules(onto) -> List[OntologyRefinement]:
    """SIDER 15개 초기 시드 규칙.

    MedDRA 온톨로지의 SOC(System Organ Class) 계층 구조를 활용하여
    장기 시스템별 독성 메커니즘을 반영.
    """
    return _collect(
        # 1. 심장 독성: hERG 차단 패턴 (LogP > 4 구간은 domain rule로)
        _domain('hasLogP', '>', 4.0),
        # 2. 간담도 독성: 방향족 아민 대사 활성화
        _fg(onto, 'AromaticAmine'),
        # 3. 위장관 독성: NSAID 계열 카복실산
        _fg(onto, 'Carboxylicacid'),
        # 4. 피부 감작성: 친전자성 Michael Acceptor
        _fg(onto, 'MichaelAcceptor'),
        # 5. 신경계 부작용: BBB 투과 + 방향족 (TPSA 저감)
        _domain('hasTPSA', '<', 90.0),
        # 6. 내분비 교란: 페놀 고리 (에스트로겐 모방)
        _fg(onto, 'Phenol'),
        # 7. 혈액/림프계: 할로겐 (알킬화제 관련)
        _fg(onto, 'Halogen'),
        # 8. 설폰아마이드: 다장기 부작용 패턴
        _fg(onto, 'Sulfonamide'),
        # 9. 아민: 기초 약리 작용기 (다수 부작용 기여)
        _fg(onto, 'Amine'),
        # 10. 방향족 분류: 다수 부작용 약물의 공통 특성
        _concept(onto, 'AromaticMolecule'),
        # 11. 분자량 과대: MW > 500 (다중 타겟 부작용)
        _domain('hasMolecularWeight', '>', 500.0),
        # 12. N+O 원자 합계: N+O > 8 (극성 과다 → 다장기 독성)
        _domain('hasNPlusOCount', '>', 8),
        # 13. 몰 굴절률: MR > 100 (거대 분자 부작용 경향)
        _domain('hasMolarRefractivity', '>', 100.0),
        # 14. 방향족 고리 과다: AromaticRings > 3
        _domain('hasNumAromaticRings', '>', 3),
        # 15. 유연성 과다: RotBonds > 8 (대사 불안정 → 부작용)
        _domain('hasNumRotatableBonds', '>', 8),
    )


# ============================================================================
# 데이터셋별 트리 하이퍼파라미터 권장값
# ============================================================================

DATASET_TREE_PARAMS: Dict[str, Dict[str, Any]] = {
    'bbbp': {
        'max_depth': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
    },
    'bace': {
        'max_depth': 6,
        'min_samples_split': 8,
        'min_samples_leaf': 4,
        'class_weight': 'balanced',
    },
    'clintox': {
        'max_depth': 4,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
    },
    'hiv': {
        'max_depth': 10,
        'min_samples_split': 15,
        'min_samples_leaf': 8,
        'class_weight': 'balanced',
    },
    'tox21': {
        'max_depth': 8,
        'min_samples_split': 12,
        'min_samples_leaf': 6,
        'class_weight': 'balanced',
    },
    'sider': {
        'max_depth': 6,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
    },
}

# 기본 하이퍼파라미터 (알 수 없는 데이터셋용)
DEFAULT_TREE_PARAMS: Dict[str, Any] = {
    'max_depth': 6,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
}


# ============================================================================
# 메인 디스패처
# ============================================================================

_RULE_GENERATORS = {
    'bbbp': get_bbbp_rules,
    'bace': get_bace_rules,
    'clintox': get_clintox_rules,
    'hiv': get_hiv_rules,
    'tox21': get_tox21_rules,
    'sider': get_sider_rules,
}


def get_dataset_seed_rules(dataset_name: str, onto) -> List[OntologyRefinement]:
    """데이터셋에 맞는 시드 규칙 리스트를 반환.

    Args:
        dataset_name: 데이터셋 이름 (대소문자 무관)
        onto: owlready2 온톨로지 인스턴스

    Returns:
        OntologyRefinement 리스트 (최대 15개)
    """
    key = str(dataset_name).strip().lower()
    generator = _RULE_GENERATORS.get(key)
    if generator is None:
        return []
    return generator(onto)


def get_dataset_tree_params(dataset_name: str) -> Dict[str, Any]:
    """데이터셋별 권장 트리 하이퍼파라미터를 반환.

    Args:
        dataset_name: 데이터셋 이름 (대소문자 무관)

    Returns:
        {'max_depth': ..., 'min_samples_split': ..., 'min_samples_leaf': ..., 'class_weight': ...}
    """
    key = str(dataset_name).strip().lower()
    return DATASET_TREE_PARAMS.get(key, DEFAULT_TREE_PARAMS).copy()
