#!/usr/bin/env python
"""
ACO-Semantic-Forest: Full Benchmark Runner
============================================
6 datasets × all targets (44 tasks) 전체 실험을 한 번에 실행한다.

Usage:
    # 전체 44개 태스크
    python experiments/run_full_benchmark.py

    # 특정 데이터셋만
    python experiments/run_full_benchmark.py --datasets tox21 sider

    # 특정 데이터셋 + 특정 타겟만
    python experiments/run_full_benchmark.py --datasets tox21 --targets NR-AhR SR-MMP

    # 샘플 수 제한 (빠른 테스트)
    python experiments/run_full_benchmark.py --max-samples 500

    # 시드 변경
    python experiments/run_full_benchmark.py --seed 123
"""

import sys
import os
import json
import time
import argparse
import traceback
import csv
import logging
import shutil
from pathlib import Path

# ── 환경 설정 ──
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

import numpy as np


def resolve_algorithm(algorithm: str) -> str:
    """Map algorithm profile name to SemanticForest criterion."""
    algo = algorithm.strip().lower()
    mapping = {
        "id3": "entropy",
        "c45": "entropy",
        "cart": "gini",
        "c5.0": "gain_ratio",
        "c50": "gain_ratio",
        "chaid": "chi_square",
        "pig": "pig",
        "semantic_similarity": "semantic_similarity",
        "semantic_sim": "semantic_similarity",
        "pig_semantic": "pig_semantic",
        "entropy": "entropy",
        "gini": "gini",
        "gain_ratio": "gain_ratio",
        "chi_square": "chi_square",
    }
    if algo not in mapping:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. "
            "Use one of: id3, c45, cart, c5.0, chaid, pig, semantic_similarity, pig_semantic, entropy, gini"
        )
    return mapping[algo]

# ── 로깅 ──
OUTPUT_DIR = PROJECT_ROOT / "output" / "benchmark_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
CSV_TIMESTAMP = time.strftime("%y%m%d_%H%M")
LOG_FILE = OUTPUT_DIR / f"benchmark_{TIMESTAMP}.log"


def resolve_output_paths(overwrite: bool, timestamp: str) -> dict:
    """Resolve output file paths based on overwrite mode."""
    aggregate_name = f"SNACO-Forest_v0.1_{CSV_TIMESTAMP}.json"
    summary_name = f"SNACO-Forest_v0.1_{CSV_TIMESTAMP}.csv"
    summary_avg_name = f"SNACO-Forest_v0.1_avg_{CSV_TIMESTAMP}.csv"

    if overwrite:
        return {
            "log": OUTPUT_DIR / "benchmark.log",
            "aggregate": OUTPUT_DIR / aggregate_name,
            "summary": OUTPUT_DIR / summary_name,
            "summary_avg": OUTPUT_DIR / summary_avg_name,
        }
    return {
        "log": OUTPUT_DIR / f"benchmark_{timestamp}.log",
        "aggregate": OUTPUT_DIR / aggregate_name,
        "summary": OUTPUT_DIR / summary_name,
        "summary_avg": OUTPUT_DIR / summary_avg_name,
    }


def clean_output_dir() -> int:
    """Delete all files/directories under output directory.

    Returns
    -------
    int
        Number of removed entries.
    """
    removed = 0
    if not OUTPUT_DIR.exists():
        return removed

    for p in OUTPUT_DIR.iterdir():
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
                removed += 1
            elif p.is_dir():
                shutil.rmtree(p)
                removed += 1
        except Exception as e:
            print(f"[WARN] Failed to remove {p}: {e}", flush=True)
    return removed


def log(msg: str) -> None:
    """Print + append to log file."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def configure_realtime_logging() -> None:
    """Enable realtime INFO logs from src.aco modules to console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("src.aco").setLevel(logging.INFO)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def parse_args():
    p = argparse.ArgumentParser(
        description="ACO-Semantic-Forest Full Benchmark Runner"
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset names to run (default: all 6)",
    )
    p.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Target names to run (only with single dataset)",
    )
    p.add_argument(
        "--ontology",
        type=str,
        default=None,
        help="Force ontology for all runs (e.g., dto, chebi, go, bao, mesh, pato, thesaurus, sio, bero, cheminf, oce, dinto, dron)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples per task (0 = unlimited)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--n-trees",
        type=int,
        default=None,
        help="Number of trees (n_estimators). Default: config default",
    )
    p.add_argument(
        "--n-ants-per-tree",
        type=int,
        default=None,
        help="Number of ants per tree per generation. Default: config default",
    )
    p.add_argument(
        "--n-generations",
        type=int,
        default=None,
        help="Number of ACO generations. Default: config default",
    )
    p.add_argument(
        "--algorithm",
        type=str,
        default="id3",
        help="Split algorithm profile: id3 | c45 | cart | c5.0 | chaid | pig | semantic_similarity | pig_semantic | entropy | gini",
    )
    p.add_argument(
        "--jump-penalty-base",
        type=float,
        default=0.0,
        help="Hierarchy jump penalty base p in [0,1] (default: 0.0)",
    )
    p.add_argument(
        "--jump-gamma",
        type=float,
        default=0.0,
        help="Hierarchy jump penalty strength gamma (default: 0.0, no penalty)",
    )
    p.add_argument(
        "--pig-alpha",
        type=float,
        default=1.0,
        help="PIG: ATI weight alpha (default: 1.0). Only effective with pig/pig_semantic criterion.",
    )
    p.add_argument(
        "--semantic-weight",
        type=float,
        default=0.3,
        help="Semantic Similarity weight in [0,1] (default: 0.3). "
             "Only effective with semantic_similarity/pig_semantic criterion.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-task JSON files (default: skip existing)",
    )
    p.add_argument(
        "--no-interpret",
        action="store_true",
        help="Skip TOP-K rule interpretation (faster)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top rules to extract (default: 5)",
    )
    return p.parse_args()


def run_benchmark(args):
    """Run the full benchmark."""
    global LOG_FILE

    if args.overwrite:
        removed = clean_output_dir()
        print(
            f"[INIT] --overwrite enabled: cleaned output directory ({removed} entries removed)",
            flush=True,
        )

    paths = resolve_output_paths(args.overwrite, TIMESTAMP)
    LOG_FILE = paths["log"]

    configure_realtime_logging()

    log("=" * 70)
    log("  ACO-Semantic-Forest: Full Benchmark")
    log("=" * 70)
    log(f"  Timestamp : {TIMESTAMP}")
    log(f"  Datasets  : {args.datasets or 'ALL'}")
    log(f"  Targets   : {args.targets or 'ALL per dataset'}")
    log(f"  Ontology  : {args.ontology or 'dataset default'}")
    log(f"  MaxSamples: {args.max_samples or 'unlimited'}")
    log(f"  Seed      : {args.seed}")
    log(f"  NTrees    : {args.n_trees or 'config default'}")
    log(f"  NAnts     : {args.n_ants_per_tree or 'config default'}")
    log(f"  NGen      : {args.n_generations or 'config default'}")
    log(f"  Algorithm : {args.algorithm}")
    log(f"  JumpBase  : {args.jump_penalty_base}")
    log(f"  JumpGamma : {args.jump_gamma}")
    log(f"  PIG Alpha : {args.pig_alpha}")
    log(f"  SemWeight : {args.semantic_weight}")
    log(f"  Overwrite : {args.overwrite}")
    log(f"  Output    : {OUTPUT_DIR}")
    log("=" * 70)

    criterion = resolve_algorithm(args.algorithm)
    if args.algorithm.lower() == "c45":
        log("  Note      : c45 uses entropy approximation in current SemanticForest")

    # ── Import pipeline ──
    log("Importing ToxicityDataPipeline...")
    from src.aco.data_pipeline import ToxicityDataPipeline

    pipeline = ToxicityDataPipeline(
        config_path="configs/dataset_ontology_config.json",
        base_dir=str(PROJECT_ROOT),
    )

    # ── Build task list ──
    if args.targets and args.datasets and len(args.datasets) == 1:
        # Specific dataset + specific targets
        tasks = [(args.datasets[0], t) for t in args.targets]
    else:
        tasks = pipeline.get_all_tasks(args.datasets)

    total = len(tasks)
    log(f"Total tasks: {total}")
    dataset_counts = {}
    for ds, _ in tasks:
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    summary = ", ".join(
        f"{ds}={cnt}" for ds, cnt in dataset_counts.items()
    )
    log(f"Task breakdown: {summary}")
    log("-" * 70)

    # ── Run ──
    all_results = []
    successes = 0
    failures = 0
    start_all = time.time()

    for idx, (ds_name, target) in enumerate(tasks, 1):
        log("")
        log(f"{'='*60}")
        log(f"  TASK {idx}/{total}: {ds_name} / {target}")
        log(f"{'='*60}")

        try:
            safe_target = target.replace(" ", "_").replace(",", "")
            json_path = OUTPUT_DIR / f"{ds_name}_{safe_target}.json"

            if json_path.exists() and not args.overwrite:
                log(f"  SKIP       : existing result found ({json_path.name})")
                with open(json_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                cached.setdefault("status", "complete")
                cached.setdefault("dataset", ds_name)
                cached.setdefault("target", target)
                all_results.append(cached)
                successes += 1
                continue

            t0 = time.time()
            result = pipeline.run(
                ds_name,
                target=target,
                ontology_override=args.ontology,
                max_samples=args.max_samples,
                n_trees=args.n_trees,
                n_ants_per_tree=args.n_ants_per_tree,
                n_generations=args.n_generations,
                criterion=criterion,
                jump_penalty_base=args.jump_penalty_base,
                jump_gamma=args.jump_gamma,
                pig_alpha=args.pig_alpha,
                semantic_weight=args.semantic_weight,
                seed=args.seed,
            )
            elapsed = time.time() - t0

            # ── Interpret top rules ──
            top_rules = []
            if not args.no_interpret and result.get("status") == "complete":
                try:
                    # Access the most recently trained forest
                    # We re-run interpret by accessing the forest from result
                    # Note: The pipeline returns metrics but we can extract
                    # top_rules from the result's feature_importance
                    log(f"  Top features by importance:")
                    fi = result.get("feature_importance", {})
                    for rank, (feat, score) in enumerate(
                        sorted(fi.items(), key=lambda x: -x[1])[:args.top_k], 1
                    ):
                        log(f"    {rank}. {feat}: {score:.6f}")
                except Exception as e:
                    log(f"  (interpret skipped: {e})")

            # ── Log metrics ──
            log(f"  Status     : {result.get('status', 'unknown')}")
            log(f"  Samples    : {result.get('n_samples', '?')}")
            log(f"  Train/Test : {result.get('n_train', '?')}/{result.get('n_test', '?')}")
            log(f"  Features   : {result.get('n_features', '?')}")
            log(f"  Trees      : {result.get('n_trees', '?')}")
            log(f"  Rules      : {result.get('n_rules', '?')}")
            log(f"  Fit time   : {result.get('fit_time_sec', 0):.1f}s")

            auc = result.get("test_auc_roc", -1)
            f1 = result.get("test_f1_majority", 0)
            bal_acc = result.get("test_balanced_accuracy_majority", 0)
            prec = result.get("test_precision_majority", 0)
            rec = result.get("test_recall_majority", 0)

            log(f"  AUC-ROC    : {auc:.4f}")
            log(f"  F1         : {f1:.4f}")
            log(f"  BalAcc     : {bal_acc:.4f}")
            log(f"  Precision  : {prec:.4f}")
            log(f"  Recall     : {rec:.4f}")
            log(f"  Pheromone  : {result.get('pheromone_sparsity_pct', 0):.2f}% active")
            log(f"  Total time : {elapsed:.1f}s")

            # ── Save per-task JSON ──
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, default=str)

            all_results.append(result)
            successes += 1

        except Exception as e:
            elapsed = time.time() - t0 if "t0" in dir() else 0
            log(f"  ERROR: {e}")
            log(traceback.format_exc())
            err_result = {
                "dataset": ds_name,
                "target": target,
                "status": "error",
                "error": str(e),
                "total_time_sec": elapsed,
            }
            all_results.append(err_result)
            failures += 1

    # ── Summary ──
    total_time = time.time() - start_all

    log("")
    log("=" * 70)
    log("  BENCHMARK SUMMARY")
    log("=" * 70)
    log(f"  Total tasks  : {total}")
    log(f"  Successes    : {successes}")
    log(f"  Failures     : {failures}")
    log(f"  Total time   : {total_time:.1f}s ({total_time/60:.1f}min)")
    log("")

    # ── Summary table ──
    header = f"{'Dataset':12s} {'Target':35s} {'AUC-ROC':>8s} {'F1':>8s} {'BalAcc':>8s} {'Prec':>8s} {'Rec':>8s} {'Time':>7s} {'Status':>8s}"
    sep = "-" * len(header)
    log(header)
    log(sep)

    for r in all_results:
        ds = r.get("dataset", "?")
        tgt = r.get("target", "?")
        status = r.get("status", "?")
        if status == "complete":
            auc = r.get("test_auc_roc", -1)
            f1 = r.get("test_f1_majority", 0)
            ba = r.get("test_balanced_accuracy_majority", 0)
            pr = r.get("test_precision_majority", 0)
            rc = r.get("test_recall_majority", 0)
            tm = r.get("total_time_sec", 0)
            log(f"{ds:12s} {tgt:35s} {auc:8.4f} {f1:8.4f} {ba:8.4f} {pr:8.4f} {rc:8.4f} {tm:6.1f}s {'OK':>8s}")
        else:
            err = r.get("error", "unknown")[:30]
            log(f"{ds:12s} {tgt:35s} {'---':>8s} {'---':>8s} {'---':>8s} {'---':>8s} {'---':>8s} {'---':>7s} {'FAIL':>8s}")

    log(sep)

    # ── Aggregate metrics (successful only) ──
    completed = [r for r in all_results if r.get("status") == "complete"]
    if completed:
        aucs = [r["test_auc_roc"] for r in completed if r.get("test_auc_roc", -1) > 0]
        f1s = [r["test_f1_majority"] for r in completed]
        bas = [r["test_balanced_accuracy_majority"] for r in completed]

        log("")
        log(f"  Mean AUC-ROC   : {np.mean(aucs):.4f} (std={np.std(aucs):.4f}, n={len(aucs)})")
        log(f"  Mean F1        : {np.mean(f1s):.4f} (std={np.std(f1s):.4f})")
        log(f"  Mean BalAcc    : {np.mean(bas):.4f} (std={np.std(bas):.4f})")

    # ── Save aggregate JSON ──
    agg_path = paths["aggregate"]
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": TIMESTAMP,
                "seed": args.seed,
                "max_samples": args.max_samples,
                "total_tasks": total,
                "successes": successes,
                "failures": failures,
                "total_time_sec": total_time,
                "results": all_results,
            },
            f,
            indent=2,
            default=str,
        )
    log(f"\n  Aggregate JSON : {agg_path}")

    # ── Save summary CSV ──
    csv_path = paths["summary"]
    csv_fields = [
        "dataset", "test_auc_roc", "target", "status",
        "n_samples", "n_train", "n_test", "n_features",
        "n_trees", "n_rules", "fit_time_sec", "total_time_sec",
        "test_f1_majority", "test_balanced_accuracy_majority",
        "test_precision_majority", "test_recall_majority",
        "test_accuracy_majority", "test_f1_weighted",
        "train_accuracy", "train_f1",
        "pheromone_sparsity_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)
    log(f"  Summary CSV    : {csv_path}")

    # ── Save average summary CSV (overall + per-dataset) ──
    avg_csv_path = paths["summary_avg"]
    avg_fields = [
        "scope",
        "dataset",
        "mean_test_auc_roc",
        "n_tasks",
        "n_completed",
        "n_failed",
        "mean_test_f1_majority",
        "mean_test_balanced_accuracy_majority",
        "mean_test_precision_majority",
        "mean_test_recall_majority",
        "mean_test_accuracy_majority",
        "mean_test_f1_weighted",
        "mean_fit_time_sec",
        "mean_total_time_sec",
    ]

    def _safe_mean(vals):
        arr = [
            float(v)
            for v in vals
            if v is not None and np.isfinite(float(v))
        ]
        return float(np.mean(arr)) if arr else float("nan")

    def _make_avg_row(scope_name, dataset_name, rows):
        completed_rows = [r for r in rows if r.get("status") == "complete"]
        auc_vals = [
            float(r.get("test_auc_roc", -1))
            for r in completed_rows
            if float(r.get("test_auc_roc", -1)) >= 0
        ]
        return {
            "scope": scope_name,
            "dataset": dataset_name,
            "n_tasks": len(rows),
            "n_completed": len(completed_rows),
            "n_failed": len(rows) - len(completed_rows),
            "mean_test_auc_roc": _safe_mean(auc_vals),
            "mean_test_f1_majority": _safe_mean([r.get("test_f1_majority") for r in completed_rows]),
            "mean_test_balanced_accuracy_majority": _safe_mean([r.get("test_balanced_accuracy_majority") for r in completed_rows]),
            "mean_test_precision_majority": _safe_mean([r.get("test_precision_majority") for r in completed_rows]),
            "mean_test_recall_majority": _safe_mean([r.get("test_recall_majority") for r in completed_rows]),
            "mean_test_accuracy_majority": _safe_mean([r.get("test_accuracy_majority") for r in completed_rows]),
            "mean_test_f1_weighted": _safe_mean([r.get("test_f1_weighted") for r in completed_rows]),
            "mean_fit_time_sec": _safe_mean([r.get("fit_time_sec") for r in completed_rows]),
            "mean_total_time_sec": _safe_mean([r.get("total_time_sec") for r in completed_rows]),
        }

    avg_rows = [_make_avg_row("overall_task_mean", "ALL", all_results)]

    grouped = {}
    for r in all_results:
        ds = r.get("dataset", "unknown")
        grouped.setdefault(ds, []).append(r)

    dataset_avg_rows = []
    for ds in sorted(grouped.keys()):
        dataset_avg_rows.append(_make_avg_row("dataset", ds, grouped[ds]))

    avg_rows.extend(dataset_avg_rows)

    if dataset_avg_rows:
        overall_dataset_row = {
            "scope": "overall_dataset_mean",
            "dataset": "ALL_DATASETS",
            "n_tasks": len(dataset_avg_rows),
            "n_completed": sum(
                1 for r in dataset_avg_rows if int(r.get("n_completed", 0)) > 0
            ),
            "n_failed": sum(
                1 for r in dataset_avg_rows if int(r.get("n_completed", 0)) == 0
            ),
            "mean_test_auc_roc": _safe_mean(
                [r.get("mean_test_auc_roc") for r in dataset_avg_rows]
            ),
            "mean_test_f1_majority": _safe_mean(
                [r.get("mean_test_f1_majority") for r in dataset_avg_rows]
            ),
            "mean_test_balanced_accuracy_majority": _safe_mean(
                [r.get("mean_test_balanced_accuracy_majority") for r in dataset_avg_rows]
            ),
            "mean_test_precision_majority": _safe_mean(
                [r.get("mean_test_precision_majority") for r in dataset_avg_rows]
            ),
            "mean_test_recall_majority": _safe_mean(
                [r.get("mean_test_recall_majority") for r in dataset_avg_rows]
            ),
            "mean_test_accuracy_majority": _safe_mean(
                [r.get("mean_test_accuracy_majority") for r in dataset_avg_rows]
            ),
            "mean_test_f1_weighted": _safe_mean(
                [r.get("mean_test_f1_weighted") for r in dataset_avg_rows]
            ),
            "mean_fit_time_sec": _safe_mean(
                [r.get("mean_fit_time_sec") for r in dataset_avg_rows]
            ),
            "mean_total_time_sec": _safe_mean(
                [r.get("mean_total_time_sec") for r in dataset_avg_rows]
            ),
        }
        avg_rows.insert(1, overall_dataset_row)

    with open(avg_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=avg_fields, extrasaction="ignore")
        writer.writeheader()
        for row in avg_rows:
            writer.writerow(row)
    log(f"  Summary AVG CSV: {avg_csv_path}")

    log(f"\n  Log file       : {LOG_FILE}")
    log("=" * 70)
    log("  DONE")
    log("=" * 70)

    return all_results


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
