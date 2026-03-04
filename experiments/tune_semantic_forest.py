# flake8: noqa

"""Two-stage random hyperparameter tuning for Semantic Forest.

Current constraints:
- Metric fixed to AUC
- Seed fixed to 42 (enforced by runner/evaluator)

Stage 1: tree hyperparameters (algorithm, n_estimators, depth, split, leaf)
Stage 2: ontology refinement profile parameters

Outputs:
- output/tuning/tuning_trials.csv
- output/tuning/best_params.yaml
"""

import argparse
import csv
import copy
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _read_best_auc(csv_path: Path) -> float:
    if not csv_path.exists():
        return float("nan")
    best = float("nan")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                auc = float(row.get("auc", "nan"))
            except Exception:
                continue
            if auc != auc:  # nan
                continue
            if best != best or auc > best:
                best = auc
    return best


def _run_trial(cmd: List[str]) -> int:
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def _base_cmd(
    dataset_key: str,
    out_csv: Path,
    dl_config_path: Path,
    feature_cache_dir: str,
    sample_size: int,
    test_size: float,
    compute_backend: str,
    torch_device: str,
) -> List[str]:
    return [
        sys.executable,
        "experiments/verify_semantic_forest_multi.py",
        "--datasets",
        dataset_key,
        "--overwrite",
        "--out",
        str(out_csv),
        "--dl-config",
        str(dl_config_path),
        "--sample-size",
        str(sample_size),
        "--test-size",
        str(test_size),
        "--compute-backend",
        str(compute_backend),
        "--torch-device",
        str(torch_device),
        "--feature-cache-dir",
        str(feature_cache_dir),
    ]


def _sample_stage1_params(rng: random.Random) -> Dict[str, Any]:
    return {
        "algorithm": rng.choice(["id3", "c45", "cart"]),
        "n_estimators": rng.randint(5, 200),
        "max_depth": rng.randint(4, 12),
        "min_samples_split": rng.randint(4, 30),
        "min_samples_leaf": rng.randint(2, 15),
    }


def _sample_stage2_ontology(rng: random.Random, dataset_key: str, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    profile_all = dict(base_cfg.get("dataset_refinement_profile", {}))
    ds_profile = dict(profile_all.get(dataset_key, {}))

    base_ref_types = ds_profile.get("allowed_ref_types") or ["concept", "cardinality", "domain", "qualification"]
    base_ref_types = [str(x) for x in base_ref_types]

    # conjunction on/off
    if rng.random() < 0.5:
        allowed_ref_types = [x for x in base_ref_types if x != "conjunction"]
    else:
        allowed_ref_types = list(dict.fromkeys(base_ref_types + ["conjunction"]))

    full_data_props = ds_profile.get("allowed_data_properties") or [
        "hasMolecularWeight", "hasLogP", "hasTPSA", "hasNumHBA", "hasNumHBD"
    ]
    full_data_props = [str(x) for x in full_data_props]

    core_candidates = [
        "hasMolecularWeight", "hasLogP", "hasTPSA", "hasNumHBA", "hasNumHBD", "hasNumRotatableBonds"
    ]
    core_data_props = [p for p in core_candidates if p in full_data_props]
    if not core_data_props:
        core_data_props = full_data_props[: min(6, len(full_data_props))]

    data_mode = rng.choice(["core", "full"])
    chosen_data_props = core_data_props if data_mode == "core" else full_data_props

    max_q = rng.randint(12, 48)

    return {
        "allowed_ref_types": allowed_ref_types,
        "allowed_data_properties": chosen_data_props,
        "max_qualification_concepts_per_property": max_q,
        "_meta_data_mode": data_mode,
    }


def _apply_trial_config(
    base_cfg: Dict[str, Any],
    dataset_key: str,
    tree_params: Dict[str, Any],
    onto_params: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    # tree params in dataset_tree_params
    dtp = dict(cfg.get("dataset_tree_params", {}))
    dtp[dataset_key] = {
        "max_depth": int(tree_params["max_depth"]),
        "min_samples_split": int(tree_params["min_samples_split"]),
        "min_samples_leaf": int(tree_params["min_samples_leaf"]),
        "class_weight": "balanced",
    }
    cfg["dataset_tree_params"] = dtp

    # ontology profile params
    drp = dict(cfg.get("dataset_refinement_profile", {}))
    ds_profile = dict(drp.get(dataset_key, {}))
    ds_profile["allowed_ref_types"] = list(onto_params["allowed_ref_types"])
    ds_profile["allowed_data_properties"] = list(onto_params["allowed_data_properties"])
    ds_profile["max_qualification_concepts_per_property"] = int(
        onto_params["max_qualification_concepts_per_property"]
    )
    drp[dataset_key] = ds_profile
    cfg["dataset_refinement_profile"] = drp

    return cfg


def _append_trial_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = [
        "dataset",
        "stage",
        "trial",
        "algorithm",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "allowed_ref_types",
        "allowed_data_properties",
        "max_qualification_concepts_per_property",
        "auc",
        "return_code",
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def _score_item(item: Dict[str, Any]) -> float:
    auc = item.get("auc", float("nan"))
    if auc != auc:
        return -1.0
    return float(auc)


def run_tuning(args) -> int:
    rng = random.Random(42)

    base_cfg_path = Path(args.dl_config)
    base_cfg = _load_yaml(base_cfg_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = out_dir / "tuning_trials.csv"
    best_yaml = out_dir / "best_params.yaml"

    dataset_list = [d.strip().lower() for d in str(args.datasets).split(",") if d.strip()]
    if not dataset_list:
        raise ValueError("--datasets must contain at least one dataset key")

    final_best: Dict[str, Any] = {
        "metric": "auc",
        "seed": 42,
        "datasets": {},
    }

    for dataset_key in dataset_list:
        print(f"\n[TUNE] Dataset={dataset_key} | Stage1 trials={args.stage1_trials} | Stage2 trials={args.stage2_trials}", flush=True)

        # ---------------- Stage 1: Tree params ----------------
        stage1_results: List[Dict[str, Any]] = []
        for t in range(1, args.stage1_trials + 1):
            tree_params = _sample_stage1_params(rng)
            onto_params = {
                "allowed_ref_types": (base_cfg.get("dataset_refinement_profile", {}).get(dataset_key, {}).get("allowed_ref_types") or ["concept", "cardinality", "domain", "qualification"]),
                "allowed_data_properties": (base_cfg.get("dataset_refinement_profile", {}).get(dataset_key, {}).get("allowed_data_properties") or ["hasMolecularWeight", "hasLogP", "hasTPSA"]),
                "max_qualification_concepts_per_property": int((base_cfg.get("dataset_refinement_profile", {}).get(dataset_key, {}).get("max_qualification_concepts_per_property") or 24)),
            }

            cfg_trial = _apply_trial_config(base_cfg, dataset_key, tree_params, onto_params)

            with tempfile.TemporaryDirectory(prefix=f"tune_{dataset_key}_s1_") as td:
                td_path = Path(td)
                cfg_path = td_path / "dl_config_trial.yaml"
                out_csv = td_path / "trial_result.csv"
                _save_yaml(cfg_path, cfg_trial)

                cmd = _base_cmd(
                    dataset_key=dataset_key,
                    out_csv=out_csv,
                    dl_config_path=cfg_path,
                    feature_cache_dir=args.feature_cache_dir,
                    sample_size=args.sample_size,
                    test_size=args.test_size,
                    compute_backend=args.compute_backend,
                    torch_device=args.torch_device,
                )
                cmd.extend([
                    "--algorithm", tree_params["algorithm"],
                    "--n-estimators", str(tree_params["n_estimators"]),
                    "--max-depth", str(tree_params["max_depth"]),
                    "--min-samples-split", str(tree_params["min_samples_split"]),
                    "--min-samples-leaf", str(tree_params["min_samples_leaf"]),
                ])
                if args.all_tasks:
                    cmd.append("--all-tasks")

                rc = _run_trial(cmd)
                auc = _read_best_auc(out_csv)

            row = {
                "dataset": dataset_key,
                "stage": "stage1",
                "trial": t,
                "algorithm": tree_params["algorithm"],
                "n_estimators": tree_params["n_estimators"],
                "max_depth": tree_params["max_depth"],
                "min_samples_split": tree_params["min_samples_split"],
                "min_samples_leaf": tree_params["min_samples_leaf"],
                "allowed_ref_types": "",
                "allowed_data_properties": "",
                "max_qualification_concepts_per_property": "",
                "auc": auc,
                "return_code": rc,
            }
            _append_trial_row(trials_csv, row)
            stage1_results.append(dict(row))

            if t % max(1, args.progress_every) == 0:
                current_best = max(stage1_results, key=_score_item)
                print(
                    f"[TUNE][{dataset_key}][stage1] trial={t}/{args.stage1_trials} "
                    f"best_auc={current_best.get('auc', float('nan')):.4f}",
                    flush=True,
                )

        stage1_top = sorted(stage1_results, key=_score_item, reverse=True)[: max(1, args.stage1_top_k)]

        # ---------------- Stage 2: Ontology params ----------------
        stage2_results: List[Dict[str, Any]] = []
        for t in range(1, args.stage2_trials + 1):
            base_pick = rng.choice(stage1_top)
            tree_params = {
                "algorithm": base_pick["algorithm"],
                "n_estimators": int(base_pick["n_estimators"]),
                "max_depth": int(base_pick["max_depth"]),
                "min_samples_split": int(base_pick["min_samples_split"]),
                "min_samples_leaf": int(base_pick["min_samples_leaf"]),
            }
            onto_params = _sample_stage2_ontology(rng, dataset_key, base_cfg)

            cfg_trial = _apply_trial_config(base_cfg, dataset_key, tree_params, onto_params)

            with tempfile.TemporaryDirectory(prefix=f"tune_{dataset_key}_s2_") as td:
                td_path = Path(td)
                cfg_path = td_path / "dl_config_trial.yaml"
                out_csv = td_path / "trial_result.csv"
                _save_yaml(cfg_path, cfg_trial)

                cmd = _base_cmd(
                    dataset_key=dataset_key,
                    out_csv=out_csv,
                    dl_config_path=cfg_path,
                    feature_cache_dir=args.feature_cache_dir,
                    sample_size=args.sample_size,
                    test_size=args.test_size,
                    compute_backend=args.compute_backend,
                    torch_device=args.torch_device,
                )
                cmd.extend([
                    "--algorithm", tree_params["algorithm"],
                    "--n-estimators", str(tree_params["n_estimators"]),
                    "--max-depth", str(tree_params["max_depth"]),
                    "--min-samples-split", str(tree_params["min_samples_split"]),
                    "--min-samples-leaf", str(tree_params["min_samples_leaf"]),
                ])
                if args.all_tasks:
                    cmd.append("--all-tasks")

                rc = _run_trial(cmd)
                auc = _read_best_auc(out_csv)

            row = {
                "dataset": dataset_key,
                "stage": "stage2",
                "trial": t,
                "algorithm": tree_params["algorithm"],
                "n_estimators": tree_params["n_estimators"],
                "max_depth": tree_params["max_depth"],
                "min_samples_split": tree_params["min_samples_split"],
                "min_samples_leaf": tree_params["min_samples_leaf"],
                "allowed_ref_types": "|".join(onto_params["allowed_ref_types"]),
                "allowed_data_properties": "|".join(onto_params["allowed_data_properties"]),
                "max_qualification_concepts_per_property": onto_params["max_qualification_concepts_per_property"],
                "auc": auc,
                "return_code": rc,
            }
            _append_trial_row(trials_csv, row)
            stage2_results.append(dict(row))

            if t % max(1, args.progress_every) == 0:
                current_best = max(stage2_results, key=_score_item)
                print(
                    f"[TUNE][{dataset_key}][stage2] trial={t}/{args.stage2_trials} "
                    f"best_auc={current_best.get('auc', float('nan')):.4f}",
                    flush=True,
                )

        all_results = stage1_results + stage2_results
        best = max(all_results, key=_score_item)
        print(
            f"[TUNE][{dataset_key}] best_auc={best.get('auc', float('nan')):.4f} "
            f"algo={best['algorithm']} n={best['n_estimators']} depth={best['max_depth']} "
            f"split={best['min_samples_split']} leaf={best['min_samples_leaf']}",
            flush=True,
        )

        final_best["datasets"][dataset_key] = {
            "best_auc": best.get("auc", float("nan")),
            "algorithm": best["algorithm"],
            "n_estimators": int(best["n_estimators"]),
            "tree_params": {
                "max_depth": int(best["max_depth"]),
                "min_samples_split": int(best["min_samples_split"]),
                "min_samples_leaf": int(best["min_samples_leaf"]),
                "class_weight": "balanced",
            },
            "ontology_params": {
                "allowed_ref_types": [x for x in str(best.get("allowed_ref_types", "")).split("|") if x],
                "allowed_data_properties": [x for x in str(best.get("allowed_data_properties", "")).split("|") if x],
                "max_qualification_concepts_per_property": int(best.get("max_qualification_concepts_per_property") or 24),
            },
        }

    _save_yaml(best_yaml, final_best)
    print(f"\n[TUNE] Trials saved: {trials_csv}")
    print(f"[TUNE] Best params saved: {best_yaml}")

    if args.apply_best:
        target_cfg = Path(args.apply_config)
        cfg = _load_yaml(target_cfg)
        datasets = final_best.get("datasets", {})
        if isinstance(datasets, dict) and datasets:
            tree_all = dict(cfg.get("dataset_tree_params", {}))
            ref_all = dict(cfg.get("dataset_refinement_profile", {}))

            for ds_key, ds_info in datasets.items():
                if not isinstance(ds_info, dict):
                    continue
                tree_params = dict(ds_info.get("tree_params", {}))
                onto_params = dict(ds_info.get("ontology_params", {}))

                prev_tree = dict(tree_all.get(ds_key, {}))
                prev_tree.update({
                    "max_depth": int(tree_params.get("max_depth", prev_tree.get("max_depth", 6))),
                    "min_samples_split": int(tree_params.get("min_samples_split", prev_tree.get("min_samples_split", 10))),
                    "min_samples_leaf": int(tree_params.get("min_samples_leaf", prev_tree.get("min_samples_leaf", 5))),
                    "class_weight": str(tree_params.get("class_weight", prev_tree.get("class_weight", "balanced"))),
                })
                tree_all[ds_key] = prev_tree

                prev_ref = dict(ref_all.get(ds_key, {}))
                ar = onto_params.get("allowed_ref_types")
                ad = onto_params.get("allowed_data_properties")
                mq = onto_params.get("max_qualification_concepts_per_property")
                if isinstance(ar, list) and ar:
                    prev_ref["allowed_ref_types"] = [str(x) for x in ar]
                if isinstance(ad, list) and ad:
                    prev_ref["allowed_data_properties"] = [str(x) for x in ad]
                if mq is not None:
                    prev_ref["max_qualification_concepts_per_property"] = int(mq)
                ref_all[ds_key] = prev_ref

            cfg["dataset_tree_params"] = tree_all
            cfg["dataset_refinement_profile"] = ref_all
            _save_yaml(target_cfg, cfg)
            print(f"[TUNE] Applied best params to config: {target_cfg}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Two-stage random tuner for Semantic Forest (AUC only, seed fixed=42)")
    p.add_argument("--datasets", required=True, help="Comma-separated dataset keys (e.g., bbbp,bace,clintox)")
    p.add_argument("--dl-config", default=str(Path("configs") / "dl_reasoner_config.yaml"))
    p.add_argument("--out-dir", default=str(Path("output") / "tuning"))
    p.add_argument("--stage1-trials", type=int, default=60)
    p.add_argument("--stage1-top-k", type=int, default=10)
    p.add_argument("--stage2-trials", type=int, default=120)
    p.add_argument("--sample-size", type=int, default=0)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--all-tasks", action="store_true", help="Tune with all tasks for multi-task datasets")
    p.add_argument("--feature-cache-dir", default=str(Path("output") / "feature_cache"))
    p.add_argument("--compute-backend", default="auto", choices=["auto", "numpy", "torch"])
    p.add_argument("--torch-device", default="auto")
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument("--apply-best", action="store_true", help="Apply best params to config after tuning")
    p.add_argument("--apply-config", default=str(Path("configs") / "dl_reasoner_config.yaml"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run_tuning(args))


if __name__ == "__main__":
    main()
