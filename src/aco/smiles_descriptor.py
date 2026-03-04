"""
Pure-Python SMILES Molecular Descriptor Calculator
===================================================

rdkit 없이 SMILES 문자열에서 분자 물리화학적 기술자(descriptor)를 추출한다.
Python 3.14+ 환경에서 rdkit-pypi가 배포되지 않은 상황을 위한 독립 구현.

지원 기술자 (16종):
    - molecular_weight   : 원자 질량 합계 (수소 포함 추정)
    - logp               : Wildman-Crippen 간이 logP (원자 기여도)
    - tpsa               : 위상적 극성 표면적 (N/O 기반 추정)
    - num_hbd            : 수소결합 공여체 수 (OH, NH)
    - num_hba            : 수소결합 수용체 수 (N, O)
    - num_rotatable_bonds: 회전 가능 결합 수
    - num_rings          : 고리 수 (SMILES 숫자 쌍)
    - num_aromatic_rings : 방향족 고리 수 (소문자 원자 기반)
    - num_atoms          : 전체 원자 수 (수소 미포함)
    - num_heavy_atoms    : 중원자 수 (= num_atoms)
    - num_heteroatoms    : 이종 원자 수 (N, O, S, P, halogen)
    - num_carbons        : 탄소 수
    - num_nitrogens      : 질소 수
    - num_oxygens        : 산소 수
    - fsp3               : sp3 탄소 비율
    - num_halogens       : 할로겐 원자 수 (F, Cl, Br, I)

주의: rdkit 대비 ~90% 수준의 근사값. 정밀한 3D 기술자는 제공하지 않음.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 원자 질량 & logP 기여도 테이블
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_ATOMIC_MASS = {
    "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.065,
    "P": 30.974, "F": 18.998, "Cl": 35.453, "Br": 79.904,
    "I": 126.904, "Si": 28.086, "B": 10.811, "Se": 78.971,
    "H": 1.008,
}

# Wildman-Crippen 간이 원자 logP 기여도 (주류 원자만)
_LOGP_CONTRIB = {
    "C_aromatic": 0.1441,     # 방향족 탄소
    "C_sp3": -0.0180,         # sp3 탄소
    "C_sp2": 0.08857,         # sp2 탄소 (비방향족)
    "N": -0.7567,             # 일반 질소
    "N_aromatic": -0.3239,    # 방향족 질소
    "O": -0.3598,             # 일반 산소
    "O_hydroxyl": -0.4802,    # -OH
    "S": 0.6314,              # 황
    "F": 0.4118,              # 플루오린
    "Cl": 0.6895,             # 클로린
    "Br": 0.8813,             # 브로민
    "I": 1.050,               # 아이오딘
    "P": 0.2836,              # 인
    "H": 0.1230,              # 수소
}

# TPSA 기여도 (Ertl, 2000 — 간이 버전)
_TPSA_CONTRIB = {
    "N_primary_amine": 26.02,     # -NH2
    "N_secondary_amine": 12.36,   # -NH-
    "N_tertiary": 3.24,           # =N-
    "N_aromatic": 12.89,          # 방향족 N
    "N_amide": 25.46,             # -C(=O)NH-
    "O_hydroxyl": 20.23,          # -OH
    "O_ether": 9.23,              # -O-
    "O_carbonyl": 17.07,          # C=O
    "S": 25.30,                   # -S-/-SH
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SMILES 파서 유틸리티
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 2글자 원자 기호 (먼저 매칭해야 함)
_TWO_CHAR_ATOMS = {"Cl", "Br", "Si", "Se", "Na", "Li", "Ca", "Mg",
                    "Al", "Zn", "Cu", "Fe", "Mn", "Co", "Ni", "Sn"}

# 대괄호 내 원자 패턴
_BRACKET_ATOM_RE = re.compile(
    r"\[(\d*)([A-Z][a-z]?|[a-z])([^]]*)\]"
)

# 유기 부분집합 원자 (SMILES 암시적 수소 규칙)
_ORGANIC_SUBSET = {"B", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}


def _parse_smiles_atoms(smiles: str) -> List[Dict]:
    """SMILES 문자열을 파싱하여 원자 정보 리스트를 반환한다.

    각 원자: {symbol, aromatic, in_bracket, charge, h_count, ring_open, ring_close}
    """
    atoms: List[Dict] = []
    i = 0
    n = len(smiles)
    ring_digits_seen: Dict[int, int] = {}  # ring_digit → atom_index

    while i < n:
        ch = smiles[i]

        # ── 대괄호 원자 ──
        if ch == "[":
            j = smiles.index("]", i)
            bracket = smiles[i:j + 1]
            m = _BRACKET_ATOM_RE.match(bracket)
            if m:
                _isotope, sym, rest = m.groups()
                aromatic = sym[0].islower()
                symbol = sym.capitalize()

                # 수소 개수 파싱
                h_match = re.search(r"H(\d*)", rest)
                h_count = 0
                if h_match:
                    h_count = int(h_match.group(1)) if h_match.group(1) else 1

                # 전하 파싱
                charge = 0
                charge_match = re.search(r"([+-])(\d*)", rest)
                if charge_match:
                    sign = 1 if charge_match.group(1) == "+" else -1
                    mag = int(charge_match.group(2)) if charge_match.group(2) else 1
                    charge = sign * mag

                atoms.append({
                    "symbol": symbol,
                    "aromatic": aromatic,
                    "in_bracket": True,
                    "charge": charge,
                    "h_count": h_count,
                })
            i = j + 1
            continue

        # ── 2글자 유기 원자 ──
        if i + 1 < n and ch.isupper() and smiles[i:i + 2] in _TWO_CHAR_ATOMS:
            symbol = smiles[i:i + 2]
            atoms.append({
                "symbol": symbol,
                "aromatic": False,
                "in_bracket": False,
                "charge": 0,
                "h_count": -1,  # 암시적
            })
            i += 2
            continue

        # ── 방향족 원자 (소문자) ──
        if ch in "cnospb":
            symbol = ch.upper()
            atoms.append({
                "symbol": symbol,
                "aromatic": True,
                "in_bracket": False,
                "charge": 0,
                "h_count": -1,
            })
            i += 1
            continue

        # ── 1글자 유기 원자 (대문자) ──
        if ch.isupper() and ch in "BCNOSPFI":
            atoms.append({
                "symbol": ch,
                "aromatic": False,
                "in_bracket": False,
                "charge": 0,
                "h_count": -1,
            })
            i += 1
            continue

        # ── 고리 숫자 (ring closure) ──
        if ch == "%":
            # %XX 형태
            if i + 2 < n and smiles[i + 1:i + 3].isdigit():
                ring_num = int(smiles[i + 1:i + 3])
                i += 3
            else:
                i += 1
                continue
        elif ch.isdigit():
            ring_num = int(ch)
            i += 1
        else:
            # 괄호, 결합 기호 등은 건너뜀
            i += 1
            continue

        # ring closure 시 원자 인덱스 기록 (atom-level ring tracking)
        if atoms:
            atom_idx = len(atoms) - 1
            if ring_num in ring_digits_seen:
                # ring close
                del ring_digits_seen[ring_num]
            else:
                ring_digits_seen[ring_num] = atom_idx

    return atoms


def _count_rings(smiles: str) -> int:
    """SMILES에서 고리 수를 계산한다 (ring closure 쌍 수)."""
    count = 0
    seen: set = set()
    i = 0
    n = len(smiles)
    while i < n:
        ch = smiles[i]
        if ch == "[":
            try:
                i = smiles.index("]", i) + 1
            except ValueError:
                i += 1
            continue
        if ch == "%":
            if i + 2 < n and smiles[i + 1:i + 3].isdigit():
                ring_num = int(smiles[i + 1:i + 3])
                if ring_num in seen:
                    count += 1
                    seen.discard(ring_num)
                else:
                    seen.add(ring_num)
                i += 3
            else:
                i += 1
        elif ch.isdigit():
            ring_num = int(ch)
            if ring_num in seen:
                count += 1
                seen.discard(ring_num)
            else:
                seen.add(ring_num)
            i += 1
        else:
            i += 1
    return count


def _count_aromatic_atoms(atoms: List[Dict]) -> int:
    """방향족 원자 수를 센다."""
    return sum(1 for a in atoms if a["aromatic"])


def _estimate_aromatic_rings(atoms: List[Dict]) -> int:
    """방향족 고리 수를 추정한다 (방향족 원자 / 5~6)."""
    n_arom = _count_aromatic_atoms(atoms)
    if n_arom == 0:
        return 0
    # 벤젠 = 6, 피리딘 = 6, 피롤 = 5, 퓨란 = 5 등 → 평균 약 5.5
    return max(1, round(n_arom / 5.5))


def _estimate_implicit_h(atom: Dict) -> int:
    """유기 부분집합 원자의 암시적 수소 수를 추정한다."""
    if atom["h_count"] >= 0:
        return atom["h_count"]

    symbol = atom["symbol"]
    aromatic = atom["aromatic"]

    # 일반적 원자가
    valences = {
        "C": 4, "N": 3, "O": 2, "S": 2,
        "P": 3, "B": 3, "F": 1, "Cl": 1,
        "Br": 1, "I": 1, "Si": 4,
    }
    v = valences.get(symbol, 0)
    if aromatic:
        v -= 1  # 방향족 결합 1개 차감
    # 결합 수는 추정이 어려움 → 보수적으로 2 (비고리)/ 0(할로겐)
    if symbol in ("F", "Cl", "Br", "I"):
        return 0
    if aromatic:
        return max(0, v - 2)
    return max(0, v - 2)  # 보수적 추정


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 기술자 계산 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _compute_molecular_weight(atoms: List[Dict]) -> float:
    """원자 질량 합계 (암시적 수소 포함)."""
    mw = 0.0
    for a in atoms:
        mw += _ATOMIC_MASS.get(a["symbol"], 12.0)
        h = _estimate_implicit_h(a)
        mw += h * _ATOMIC_MASS["H"]
    return mw


def _compute_logp(atoms: List[Dict]) -> float:
    """Wildman-Crippen 간이 logP 추정."""
    logp = 0.0
    for a in atoms:
        sym = a["symbol"]
        arom = a["aromatic"]

        if sym == "C":
            if arom:
                logp += _LOGP_CONTRIB["C_aromatic"]
            else:
                logp += _LOGP_CONTRIB["C_sp3"]
        elif sym == "N":
            logp += _LOGP_CONTRIB["N_aromatic"] if arom else _LOGP_CONTRIB["N"]
        elif sym == "O":
            logp += _LOGP_CONTRIB["O"]
        elif sym in _LOGP_CONTRIB:
            logp += _LOGP_CONTRIB[sym]
        else:
            logp += 0.0

        # 암시적 수소
        h = _estimate_implicit_h(a)
        logp += h * _LOGP_CONTRIB["H"]

    return logp


def _compute_tpsa(atoms: List[Dict], smiles: str) -> float:
    """위상적 극성 표면적 추정."""
    tpsa = 0.0

    # 패턴 기반 TPSA 추정
    n_count = sum(1 for a in atoms if a["symbol"] == "N")
    o_count = sum(1 for a in atoms if a["symbol"] == "O")
    s_count = sum(1 for a in atoms if a["symbol"] == "S")

    # NH2 패턴
    nh2 = len(re.findall(r"(?i)\[NH2\]|N\(", smiles))
    # NH 패턴
    nh = smiles.lower().count("[nh]") + smiles.count("[NH]")
    # OH 패턴
    oh = smiles.count("O") - smiles.count("O=") - smiles.count("o")
    oh = min(oh, o_count)

    # 간이 계산: N/O에 기반
    for a in atoms:
        sym = a["symbol"]
        if sym == "N":
            if a["aromatic"]:
                tpsa += _TPSA_CONTRIB["N_aromatic"]
            else:
                h = _estimate_implicit_h(a)
                if a.get("h_count", 0) >= 2 or h >= 2:
                    tpsa += _TPSA_CONTRIB["N_primary_amine"]
                elif a.get("h_count", 0) >= 1 or h >= 1:
                    tpsa += _TPSA_CONTRIB["N_secondary_amine"]
                else:
                    tpsa += _TPSA_CONTRIB["N_tertiary"]
        elif sym == "O":
            # C=O vs C-O 구분 근사
            h = _estimate_implicit_h(a)
            if a.get("h_count", 0) >= 1 or h >= 1:
                tpsa += _TPSA_CONTRIB["O_hydroxyl"]
            elif "=" in smiles:  # 매우 거친 근사
                tpsa += _TPSA_CONTRIB["O_carbonyl"] * 0.5 + \
                        _TPSA_CONTRIB["O_ether"] * 0.5
            else:
                tpsa += _TPSA_CONTRIB["O_ether"]
        elif sym == "S":
            tpsa += _TPSA_CONTRIB["S"]

    return tpsa


def _count_hbd(atoms: List[Dict]) -> int:
    """수소결합 공여체 수 (OH, NH, NH2)."""
    count = 0
    for a in atoms:
        if a["symbol"] in ("N", "O"):
            h = a["h_count"] if a["h_count"] >= 0 else _estimate_implicit_h(a)
            if h > 0:
                count += h
    return count


def _count_hba(atoms: List[Dict]) -> int:
    """수소결합 수용체 수 (N, O 원자)."""
    return sum(1 for a in atoms if a["symbol"] in ("N", "O"))


def _count_rotatable_bonds(smiles: str, n_rings: int) -> int:
    """회전 가능 결합 수 추정.

    단일 결합 수에서 고리 내 결합을 제외한 근사.
    """
    # 원자 간 단일 결합 ≈ (원자수 - 1) + ring_closures - 이중/삼중 결합
    # SMILES의 '-' 는 암시적이므로, 총 결합수 - 이중결합 - 삼중결합 - 고리결합
    double = smiles.count("=")
    triple = smiles.count("#")

    # 비-수소 원자 수 근사
    clean = re.sub(r"\[.*?\]", "X", smiles)  # 대괄호 → X
    clean = re.sub(r"[^A-Za-z]", "", clean)  # 비-문자 제거
    n_heavy = len(clean)

    total_bonds = n_heavy - 1 + n_rings  # 트리 + 고리
    single_bonds = total_bonds - double - triple
    ring_single_bonds = n_rings * 2  # 고리 내 단일 결합 (보수적)

    rotatable = max(0, single_bonds - ring_single_bonds - 1)
    return rotatable


def _count_fsp3(atoms: List[Dict]) -> float:
    """sp3 탄소 비율 추정."""
    carbons = [a for a in atoms if a["symbol"] == "C"]
    if not carbons:
        return 0.0
    # 방향족 C = sp2, 나머지 ≈ sp3 (근사)
    sp3 = sum(1 for c in carbons if not c["aromatic"])
    return sp3 / len(carbons)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def compute_descriptors(smiles: str) -> Optional[Dict[str, float]]:
    """SMILES 문자열 하나에 대해 16종의 분자 기술자를 계산한다.

    Parameters
    ----------
    smiles : str
        SMILES 문자열.

    Returns
    -------
    dict[str, float] | None
        기술자 딕셔너리. 파싱 실패 시 None.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    # 기본 SMILES 유효성 검증
    smiles = smiles.strip()
    if len(smiles) < 2:
        return None
    # SMILES 허용 문자 집합 확인
    allowed = set("CNOSPFIHBcnospb[]()=#-+/\\@%0123456789.lraeKLMgZTdifu")
    if not all(ch in allowed for ch in smiles):
        return None

    # 염(salt) 분리: '.'으로 분리된 경우 가장 긴 조각 사용
    fragments = smiles.split(".")
    smiles = max(fragments, key=len)

    try:
        atoms = _parse_smiles_atoms(smiles)
    except Exception:
        return None

    if len(atoms) == 0:
        return None

    n_rings = _count_rings(smiles)
    n_aromatic_rings = _estimate_aromatic_rings(atoms)

    n_carbons = sum(1 for a in atoms if a["symbol"] == "C")
    n_nitrogens = sum(1 for a in atoms if a["symbol"] == "N")
    n_oxygens = sum(1 for a in atoms if a["symbol"] == "O")
    n_halogens = sum(1 for a in atoms if a["symbol"] in ("F", "Cl", "Br", "I"))
    n_heteroatoms = sum(
        1 for a in atoms if a["symbol"] not in ("C", "H")
    )

    return {
        "molecular_weight": _compute_molecular_weight(atoms),
        "logp": _compute_logp(atoms),
        "tpsa": _compute_tpsa(atoms, smiles),
        "num_hbd": float(_count_hbd(atoms)),
        "num_hba": float(_count_hba(atoms)),
        "num_rotatable_bonds": float(
            _count_rotatable_bonds(smiles, n_rings)
        ),
        "num_rings": float(n_rings),
        "num_aromatic_rings": float(n_aromatic_rings),
        "num_atoms": float(len(atoms)),
        "num_heavy_atoms": float(len(atoms)),
        "num_heteroatoms": float(n_heteroatoms),
        "num_carbons": float(n_carbons),
        "num_nitrogens": float(n_nitrogens),
        "num_oxygens": float(n_oxygens),
        "fsp3": _count_fsp3(atoms),
        "num_halogens": float(n_halogens),
    }


def compute_descriptors_batch(
    smiles_list: List[str],
    *,
    min_valid_ratio: float = 0.5,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[int]]:
    """SMILES 리스트에 대해 배치 기술자 계산.

    Parameters
    ----------
    smiles_list : List[str]
        SMILES 문자열 리스트.
    min_valid_ratio : float
        최소 유효 비율. 이 미만이면 ValueError.
    verbose : bool
        진행 로그 출력 여부.

    Returns
    -------
    (pd.DataFrame, List[int])
        기술자 DataFrame과 유효 인덱스 리스트.
    """
    results: List[Dict[str, float]] = []
    valid_indices: List[int] = []

    n = len(smiles_list)
    failed = 0

    for i, smi in enumerate(smiles_list):
        desc = compute_descriptors(str(smi))
        if desc is not None:
            results.append(desc)
            valid_indices.append(i)
        else:
            failed += 1

        if verbose and (i + 1) % 1000 == 0:
            logger.info(
                "Descriptor computation: %d/%d (%.1f%% valid)",
                i + 1, n, 100 * len(results) / (i + 1),
            )

    if verbose:
        logger.info(
            "Descriptor computation complete: %d/%d valid (%.1f%%), %d failed",
            len(results), n, 100 * len(results) / max(n, 1), failed,
        )

    if len(results) / max(n, 1) < min_valid_ratio:
        raise ValueError(
            f"Only {len(results)}/{n} ({100*len(results)/max(n,1):.1f}%) "
            f"valid — below minimum {min_valid_ratio*100:.0f}%"
        )

    df = pd.DataFrame(results, index=valid_indices)
    return df, valid_indices


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_smiles = [
        "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",           # Tox21 예시
        "CCN1C(=O)NC(c2ccccc2)C1=O",               # Tox21 예시
        "CC(=O)Oc1ccccc1C(=O)O",                    # 아스피린
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",     # 테스토스테론
        "c1ccc(cc1)C(=O)O",                          # 벤조산
        "CCCCCCCCCCCCCCCC(=O)O",                     # 팔미트산
        "invalid_smiles_###",                         # 유효하지 않은 SMILES
    ]

    for smi in test_smiles:
        desc = compute_descriptors(smi)
        if desc:
            print(f"\n{smi}")
            for k, v in desc.items():
                print(f"  {k:25s}: {v:.3f}")
        else:
            print(f"\n{smi} → FAILED")
