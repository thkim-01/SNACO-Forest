# flake8: noqa

"""Apply tuning best parameters into DL config YAML.

Input:
- output/tuning/best_params.yaml (from tune_semantic_forest.py)

Output:
- updates configs/dl_reasoner_config.yaml in-place (with optional backup)
"""

import argparse
from pathlib import Path
from typing import Dict, Any

# MUST BE FIRST: Initialize paths for all imports
from _init_paths import init_paths
if not init_paths():
    raise SystemExit(1)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore
    if not path.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return data


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    import yaml  # type: ignore
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _apply(best: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    datasets = best.get("datasets", {})
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("best params file has no datasets entries")

    out = dict(config)

    tree_all = dict(out.get("dataset_tree_params", {}))
    ref_all = dict(out.get("dataset_refinement_profile", {}))

    for ds_key, ds_info in datasets.items():
        ds_key = str(ds_key).lower()
        if not isinstance(ds_info, dict):
            continue

        tree_params = ds_info.get("tree_params", {}) or {}
        onto_params = ds_info.get("ontology_params", {}) or {}

        # Merge tree params
        prev_tree = dict(tree_all.get(ds_key, {}))
        prev_tree.update({
            "max_depth": int(tree_params.get("max_depth", prev_tree.get("max_depth", 6))),
            "min_samples_split": int(tree_params.get("min_samples_split", prev_tree.get("min_samples_split", 10))),
            "min_samples_leaf": int(tree_params.get("min_samples_leaf", prev_tree.get("min_samples_leaf", 5))),
            "class_weight": str(tree_params.get("class_weight", prev_tree.get("class_weight", "balanced"))),
        })
        tree_all[ds_key] = prev_tree

        # Merge refinement profile params (keep existing allowed_object_properties etc.)
        prev_ref = dict(ref_all.get(ds_key, {}))
        allowed_ref_types = onto_params.get("allowed_ref_types")
        allowed_data_properties = onto_params.get("allowed_data_properties")
        max_q = onto_params.get("max_qualification_concepts_per_property")

        if isinstance(allowed_ref_types, list) and allowed_ref_types:
            prev_ref["allowed_ref_types"] = [str(x) for x in allowed_ref_types]
        if isinstance(allowed_data_properties, list) and allowed_data_properties:
            prev_ref["allowed_data_properties"] = [str(x) for x in allowed_data_properties]
        if max_q is not None:
            prev_ref["max_qualification_concepts_per_property"] = int(max_q)

        ref_all[ds_key] = prev_ref

    out["dataset_tree_params"] = tree_all
    out["dataset_refinement_profile"] = ref_all
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply best tuning YAML into dl_reasoner_config.yaml")
    p.add_argument("--best", default=str(Path("output") / "tuning" / "best_params.yaml"))
    p.add_argument("--config", default=str(Path("configs") / "dl_reasoner_config.yaml"))
    p.add_argument("--backup", action="store_true", help="Create .bak backup before overwrite")
    return p


def main() -> None:
    args = build_parser().parse_args()
    best_path = Path(args.best)
    config_path = Path(args.config)

    best = _load_yaml(best_path)
    config = _load_yaml(config_path)

    merged = _apply(best, config)

    if args.backup:
        backup = config_path.with_suffix(config_path.suffix + ".bak")
        backup.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[APPLY] Backup created: {backup}")

    _save_yaml(config_path, merged)
    print(f"[APPLY] Updated config: {config_path}")
    print(f"[APPLY] Source best params: {best_path}")


if __name__ == "__main__":
    main()
