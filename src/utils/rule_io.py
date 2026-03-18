"""
Rule IO Utilities — 규칙 저장, 내보내기 및 텍스트 변환 모듈.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import List

from src.aco.rule_extraction import DecisionPath

logger = logging.getLogger(__name__)

class RuleExporter:
    """DecisionPath 규칙 리스트를 다양한 포맷으로 내보낸다."""

    @staticmethod
    def _ensure_dir(filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def export_to_json(cls, rules: List[DecisionPath], filepath: str) -> None:
        """JSON 포맷으로 규칙 내보내기."""
        cls._ensure_dir(filepath)
        data = [r.to_dict() for r in rules]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info("Exported %d rules to JSON: %s", len(rules), filepath)

    @classmethod
    def export_to_csv(cls, rules: List[DecisionPath], filepath: str) -> None:
        """CSV 포맷으로 규칙 요약 정보 내보내기."""
        cls._ensure_dir(filepath)
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Rule String", "Prediction", "Coverage", "Fitness", 
                "F1 Score", "Accuracy", "Depth", "Total Info Gain"
            ])
            for r in rules:
                writer.writerow([
                    r.to_rule_string(),
                    r.prediction,
                    r.coverage,
                    f"{r.fitness:.4f}",
                    f"{r.f1_score:.4f}",
                    f"{r.accuracy:.4f}",
                    r.depth,
                    f"{r.total_info_gain:.4f}"
                ])
        logger.info("Exported %d rules to CSV: %s", len(rules), filepath)

    @classmethod
    def export_to_markdown(cls, rules: List[DecisionPath], filepath: str) -> None:
        """Markdown 포맷으로 규칙 내보내기."""
        cls._ensure_dir(filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Decision Rules Report\n\n")
            f.write("| Rank | Rule | Prediction | Coverage | F1 Score | Fitness | Total IG |\n")
            f.write("|---|---|---|---|---|---|---|\n")
            for i, r in enumerate(rules, 1):
                f.write(
                    f"| {i} | `{r.to_rule_string()}` | {r.prediction} | {r.coverage} | "
                    f"{r.f1_score:.4f} | {r.fitness:.4f} | {r.total_info_gain:.4f} |\n"
                )
        logger.info("Exported %d rules to Markdown: %s", len(rules), filepath)

    @classmethod
    def export_to_txt(cls, rules: List[DecisionPath], filepath: str) -> None:
        """TXT 포맷으로 규칙 내보내기."""
        cls._ensure_dir(filepath)
        with open(filepath, "w", encoding="utf-8") as f:
            for i, r in enumerate(rules, 1):
                f.write(f"Rule [{i}]: {r.to_rule_string()}\n")
                f.write(f"  - Prediction: {r.prediction}\n")
                f.write(f"  - Coverage: {r.coverage}\n")
                f.write(f"  - F1 Score: {r.f1_score:.4f}\n")
                f.write(f"  - Fitness: {r.fitness:.4f}\n")
                f.write(f"  - Depth: {r.depth}\n")
                f.write("-" * 40 + "\n")
        logger.info("Exported %d rules to TXT: %s", len(rules), filepath)

    @staticmethod
    def load_fixed_rules(filepath: str) -> List[DecisionPath]:
        """JSON 파일에서 규칙 리스트로 불러오기."""
        if not os.path.exists(filepath):
            logger.warning("Fixed rules file not found: %s", filepath)
            return []
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        rules = [DecisionPath.from_dict(d) for d in data]
        logger.info("Loaded %d fixed rules from %s", len(rules), filepath)
        return rules
