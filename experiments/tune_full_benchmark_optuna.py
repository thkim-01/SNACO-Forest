#!/usr/bin/env python
"""
Optuna-based hyperparameter tuning for ACO-Semantic-Forest full benchmark.

This tuner evaluates parameter sets by running ToxicityDataPipeline tasks and
maximizing a chosen metric (default: F1).

Outputs:
- output/tuning/optuna_trials.csv
- output/tuning/optuna_best_params.json
- output/tuning/optuna_study.db (SQLite storage)

Examples:
    # quick smoke tuning
    python experiments/tune_full_benchmark_optuna.py --n-trials 10 --max-samples 500

    # specific datasets only
    python experiments/tune_full_benchmark_optuna.py --datasets bbbp bace clintox --n-trials 30

    # single dataset + selected targets
    python experiments/tune_full_benchmark_optuna.py --datasets tox21 --targets NR-AhR SR-MMP --n-trials 20
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

try:
    import optuna
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Optuna is required. Install with: pip install optuna\n"
        f"Import error: {e}"
    )

from src.aco.data_pipeline import ToxicityDataPipeline


OUTPUT_DIR = PROJECT_ROOT / "output" / "tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_algorithm(algorithm: str) -> str:
    algo = algorithm.strip().lower()
    mapping = {
        "id3": "entropy",
        "c45": "entropy",
        "cart": "gini",
        "c5.0": "gain_ratio",
        "c5": "gain_ratio",
        "chaid": "chi_square",
        "entropy": "entropy",
        "gini": "gini",
        "gain_ratio": "gain_ratio",
        "chi_square": "chi_square",
    }
    if algo not in mapping:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            "Use one of: id3, c45, cart, c5.0, chaid, entropy, gini"
        )
    return mapping[algo]


def mean_or_nan(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.mean(values))


def build_tasks(
    pipeline: ToxicityDataPipeline,
    datasets: List[str] | None,
    targets: List[str] | None,
) -> List[Tuple[str, str]]:
    if targets and datasets and len(datasets) == 1:
        return [(datasets[0], t) for t in targets]
    return pipeline.get_all_tasks(datasets)


def score_results(results: List[Dict[str, Any]], metric: str) -> Tuple[float, Dict[str, float]]:
    completed = [r for r in results if r.get("status") == "complete"]
    if not completed:
        return -1e9, {
            "mean_f1": float("nan"),
            "mean_bal_acc": float("nan"),
            "mean_auc": float("nan"),
        }

    f1s = [float(r.get("test_f1_majority", 0.0)) for r in completed]
    bals = [float(r.get("test_balanced_accuracy_majority", 0.0)) for r in completed]
    aucs = [float(r.get("test_auc_roc", -1.0)) for r in completed]
    aucs_valid = [a for a in aucs if a >= 0.0]

    mean_f1 = mean_or_nan(f1s)
    mean_bal = mean_or_nan(bals)
    mean_auc = mean_or_nan(aucs_valid)

    if metric == "f1":
        score = mean_f1
    elif metric == "balanced_accuracy":
        score = mean_bal
    elif metric == "auc":
        score = mean_auc if mean_auc == mean_auc else -1.0
    else:  # composite
        auc_term = mean_auc if mean_auc == mean_auc else mean_bal
        score = 0.6 * mean_f1 + 0.3 * mean_bal + 0.1 * auc_term

    return float(score), {
        "mean_f1": mean_f1,
        "mean_bal_acc": mean_bal,
        "mean_auc": mean_auc,
    }


def append_trial_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = [
        "trial",
        "score",
        "mean_f1",
        "mean_bal_acc",
        "mean_auc",
        "n_tasks",
        "n_completed",
        "n_failed",
        "algorithm",
        "n_trees",
        "n_ants_per_tree",
        "n_generations",
        "jump_penalty_base",
        "jump_gamma",
        "elapsed_sec",
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuner for full benchmark")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--targets", nargs="*", default=None)
    p.add_argument("--max-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--timeout-sec", type=int, default=0)
    p.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "balanced_accuracy", "auc", "composite"],
    )

    p.add_argument("--study-name", type=str, default="aco_semantic_forest")
    p.add_argument(
        "--storage",
        type=str,
        default=str(OUTPUT_DIR / "optuna_study.db"),
        help="SQLite path or full Optuna storage URL",
    )
    p.add_argument(
        "--reset-study",
        action="store_true",
        help="Delete existing study DB file if using sqlite:///...",
    )

    p.add_argument(
        "--algorithm-space",
        nargs="*",
        default=["cart", "id3", "c45", "c5.0", "chaid"],
        help="Candidate algorithms",
    )
    p.add_argument("--n-trees-min", type=int, default=8)
    p.add_argument("--n-trees-max", type=int, default=120)
    p.add_argument("--n-ants-min", type=int, default=3)
    p.add_argument("--n-ants-max", type=int, default=30)
    p.add_argument("--n-gen-min", type=int, default=1)
    p.add_argument("--n-gen-max", type=int, default=4)

    p.add_argument("--jump-base-min", type=float, default=0.85)
    p.add_argument("--jump-base-max", type=float, default=1.0)
    p.add_argument(
        "--jump-gamma-space",
        nargs="*",
        default=["0", "0.5", "1", "2", "3"],
        help="Discrete gamma candidates; supports 'inf'",
    )

    p.add_argument(
        "--trials-csv",
        type=str,
        default=str(OUTPUT_DIR / "optuna_trials.csv"),
    )
    p.add_argument(
        "--best-json",
        type=str,
        default=str(OUTPUT_DIR / "optuna_best_params.json"),
    )

    return p.parse_args()


def normalize_storage_url(storage: str) -> str:
    if storage.startswith("sqlite://"):
        return storage
    p = Path(storage)
    p.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{p.as_posix()}"


def maybe_reset_sqlite(storage_url: str, reset: bool) -> None:
    if not reset:
        return
    if not storage_url.startswith("sqlite:///"):
        return
    db_path = Path(storage_url.replace("sqlite:///", "", 1))
    if db_path.exists():
        db_path.unlink()


def parse_gamma_space(gamma_tokens: List[str]) -> List[float]:
    vals: List[float] = []
    for token in gamma_tokens:
        t = token.strip().lower()
        if t == "inf":
            vals.append(float("inf"))
        else:
            vals.append(float(t))
    # unique, keep order
    unique = []
    seen = set()
    for v in vals:
        key = "inf" if v == float("inf") else f"{v:.12g}"
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np_seed = args.seed

    storage_url = normalize_storage_url(args.storage)
    maybe_reset_sqlite(storage_url, args.reset_study)

    gamma_space = parse_gamma_space(args.jump_gamma_space)
    trials_csv = Path(args.trials_csv)
    best_json = Path(args.best_json)

    pipeline = ToxicityDataPipeline(
        config_path="configs/dataset_ontology_config.json",
        base_dir=str(PROJECT_ROOT),
    )
    tasks = build_tasks(pipeline, args.datasets, args.targets)

    if not tasks:
        raise SystemExit("No tasks to tune. Check --datasets/--targets.")

    print(f"[OPTUNA] Tasks: {len(tasks)}")
    print(f"[OPTUNA] Metric: {args.metric}")
    print(f"[OPTUNA] Study: {args.study_name}")
    print(f"[OPTUNA] Storage: {storage_url}")

    sampler = optuna.samplers.TPESampler(seed=np_seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        t0 = time.time()

        algorithm = trial.suggest_categorical("algorithm", args.algorithm_space)
        criterion = resolve_algorithm(algorithm)

        n_trees = trial.suggest_int("n_trees", args.n_trees_min, args.n_trees_max)
        n_ants = trial.suggest_int("n_ants_per_tree", args.n_ants_min, args.n_ants_max)
        n_gen = trial.suggest_int("n_generations", args.n_gen_min, args.n_gen_max)

        jump_base = trial.suggest_float(
            "jump_penalty_base", args.jump_base_min, args.jump_base_max
        )
        jump_gamma = trial.suggest_categorical("jump_gamma", gamma_space)

        results: List[Dict[str, Any]] = []
        for ds_name, target in tasks:
            try:
                res = pipeline.run(
                    ds_name,
                    target=target,
                    max_samples=args.max_samples,
                    n_trees=n_trees,
                    n_ants_per_tree=n_ants,
                    n_generations=n_gen,
                    criterion=criterion,
                    jump_penalty_base=jump_base,
                    jump_gamma=jump_gamma,
                    seed=args.seed,
                )
                results.append(res)
            except Exception as e:
                results.append(
                    {
                        "dataset": ds_name,
                        "target": target,
                        "status": "error",
                        "error": str(e),
                    }
                )

        score, stat = score_results(results, args.metric)

        completed = [r for r in results if r.get("status") == "complete"]
        failed = [r for r in results if r.get("status") != "complete"]

        elapsed = time.time() - t0
        trial.set_user_attr("mean_f1", stat["mean_f1"])
        trial.set_user_attr("mean_bal_acc", stat["mean_bal_acc"])
        trial.set_user_attr("mean_auc", stat["mean_auc"])
        trial.set_user_attr("n_tasks", len(results))
        trial.set_user_attr("n_completed", len(completed))
        trial.set_user_attr("n_failed", len(failed))
        trial.set_user_attr("elapsed_sec", elapsed)

        append_trial_csv(
            trials_csv,
            {
                "trial": trial.number,
                "score": score,
                "mean_f1": stat["mean_f1"],
                "mean_bal_acc": stat["mean_bal_acc"],
                "mean_auc": stat["mean_auc"],
                "n_tasks": len(results),
                "n_completed": len(completed),
                "n_failed": len(failed),
                "algorithm": algorithm,
                "n_trees": n_trees,
                "n_ants_per_tree": n_ants,
                "n_generations": n_gen,
                "jump_penalty_base": jump_base,
                "jump_gamma": jump_gamma,
                "elapsed_sec": elapsed,
            },
        )

        print(
            f"[TRIAL {trial.number}] score={score:.5f} "
            f"f1={stat['mean_f1']:.5f} bal={stat['mean_bal_acc']:.5f} "
            f"auc={stat['mean_auc']:.5f} "
            f"params={{algo={algorithm}, trees={n_trees}, ants={n_ants}, gen={n_gen}, "
            f"jump_base={jump_base:.3f}, gamma={jump_gamma}}}"
        )

        return score

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=(None if args.timeout_sec <= 0 else args.timeout_sec),
        show_progress_bar=False,
    )

    best = {
        "study_name": args.study_name,
        "metric": args.metric,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "n_trials": len(study.trials),
        "tasks": tasks,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "storage": storage_url,
    }

    best_json.parent.mkdir(parents=True, exist_ok=True)
    with best_json.open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2, default=str)

    print("\n[OPTUNA] Done")
    print(f"[OPTUNA] Best score: {study.best_value:.6f}")
    print(f"[OPTUNA] Best params: {study.best_params}")
    print(f"[OPTUNA] Trials CSV : {trials_csv}")
    print(f"[OPTUNA] Best JSON  : {best_json}")


if __name__ == "__main__":
    main()
