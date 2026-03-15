#!/usr/bin/env python
"""
apply_optimal_routing.py
===================================
데이터셋별 1:1 온톨로지 라우팅 맵을 기계가 찾아낸 최적 조합으로
dataset_ontology_config.json을 전면 교체한다.

최적 조합 (기계 탐색 결과):
  BBBP    → bao
  BACE    → chebi
  HIV     → thesaurus
  ClinTox → bao
  Tox21   → chmo
  SIDER   → chem2bio2rdf

변경 항목:
  - datasets.<ds>.ontology
  - datasets.<ds>.bridge_domain
  - datasets.<ds>.hierarchy_config.source_ontology  (항목 존재 시)

실행:
  python experiments/apply_optimal_routing.py
  python experiments/apply_optimal_routing.py --dry-run   # 변경 내용만 출력, 저장 안 함
  python experiments/apply_optimal_routing.py --restore   # 백업 파일로 복원
"""

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_ontology_config.json"
BACKUP_PATH = CONFIG_PATH.with_suffix(".json.bak")

# ── 최적 라우팅 테이블 ───────────────────────────────────────────
# dataset_key → (new_ontology, new_bridge_domain)
OPTIMAL_ROUTING: dict[str, tuple[str, str]] = {
    "bbbp":    ("bao",          "anchor"),
    "bace":    ("chebi",        "chebi"),
    "hiv":     ("thesaurus",    "anchor"),
    "clintox": ("bao",          "anchor"),
    "tox21":   ("chmo",         "anchor"),
    "sider":   ("chem2bio2rdf", "anchor"),
}


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n[saved] {CONFIG_PATH}")


def apply_routing(config: dict, dry_run: bool = False) -> list[dict]:
    """라우팅을 교체하고 변경 로그를 반환한다."""
    changes = []

    for ds_key, (new_ont, new_bridge) in OPTIMAL_ROUTING.items():
        if ds_key not in config.get("datasets", {}):
            print(f"[WARN]  '{ds_key}' not found in config — skipped")
            continue

        ds = config["datasets"][ds_key]
        old_ont = ds.get("ontology", "(none)")
        old_bridge = ds.get("bridge_domain", "(none)")
        old_src_ont = None

        if not dry_run:
            ds["ontology"] = new_ont
            ds["bridge_domain"] = new_bridge
            if "hierarchy_config" in ds:
                old_src_ont = ds["hierarchy_config"].get("source_ontology")
                ds["hierarchy_config"]["source_ontology"] = new_ont

        changes.append({
            "dataset": ds_key,
            "ontology": (old_ont, new_ont),
            "bridge_domain": (old_bridge, new_bridge),
            "hierarchy_source_ontology": (old_src_ont, new_ont) if old_src_ont else None,
        })

    return changes


def print_changes(changes: list[dict]) -> None:
    col = 12
    print(f"\n{'Dataset':{col}}  {'ontology (old → new)':<38}  {'bridge_domain (old → new)':<32}  hierarchy_source")
    print("-" * 110)
    for c in changes:
        ds = c["dataset"]
        ont_old, ont_new = c["ontology"]
        br_old, br_new = c["bridge_domain"]
        hier = f"{c['hierarchy_source_ontology'][0]} → {c['hierarchy_source_ontology'][1]}" if c["hierarchy_source_ontology"] else "—"
        print(f"{ds:{col}}  {ont_old:<18} → {ont_new:<16}  {br_old:<14} → {br_new:<14}  {hier}")
    print()


def restore_backup() -> None:
    if not BACKUP_PATH.exists():
        print(f"[ERROR] Backup not found: {BACKUP_PATH}")
        raise SystemExit(1)
    shutil.copy2(BACKUP_PATH, CONFIG_PATH)
    print(f"[restored] {CONFIG_PATH}  ← {BACKUP_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply optimal ontology routing to dataset_ontology_config.json")
    parser.add_argument("--dry-run", action="store_true", help="변경 내용만 출력하고 저장하지 않음")
    parser.add_argument("--restore", action="store_true", help="백업(.json.bak)으로 복원")
    args = parser.parse_args()

    if args.restore:
        restore_backup()
        return

    config = load_config()

    if not args.dry_run:
        # 기존 파일 백업
        shutil.copy2(CONFIG_PATH, BACKUP_PATH)
        print(f"[backup]  {BACKUP_PATH}")

    changes = apply_routing(config, dry_run=args.dry_run)
    print_changes(changes)

    if args.dry_run:
        print("[dry-run] 저장하지 않았습니다. --dry-run 없이 실행하면 실제로 적용됩니다.")
        return

    # 변경 사항을 검증: 모든 new_ont 값이 ontologies 섹션에 등록되어 있는지 확인
    known_onts = set(config.get("ontologies", {}).keys())
    missing = {new_ont for _, (new_ont, _) in OPTIMAL_ROUTING.items() if new_ont not in known_onts}
    if missing:
        print(f"[WARN]  다음 온톨로지가 config의 'ontologies' 섹션에 없습니다: {missing}")
        print("        동작은 하지만 OWLGraphBuilder 로딩 시 실패할 수 있습니다.")

    save_config(config)

    print(f"[info]  원본 복원: python experiments/apply_optimal_routing.py --restore")


if __name__ == "__main__":
    main()
