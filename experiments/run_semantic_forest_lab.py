# flake8: noqa

"""Run Semantic-Forest-lab with algorithm profiles (ID3/C4.5/CART).

Examples:
  python experiments/run_semantic_forest_lab.py --algorithm id3
  python experiments/run_semantic_forest_lab.py --algorithm c45 --datasets bbbp,clintox
    python experiments/run_semantic_forest_lab.py --algorithm cart --all-tasks
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# MUST BE FIRST: Initialize paths for all imports
from _init_paths import init_paths
if not init_paths():
    raise SystemExit(1)

from src.algorithms import load_algorithm_profile

# Set PYTHONPATH for subprocess calls
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')

FIXED_RANDOM_STATE = 42


def build_command(args, profile: dict):
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    key = profile["algorithm_key"]
    out_csv = out_base / f"semantic_forest_benchmark_{key}.csv"

    cmd = [
        sys.executable,
        "experiments/verify_semantic_forest_multi.py",
        "--split-criterion",
        str(profile["split_criterion"]),
        "--n-estimators",
        str(args.n_estimators if args.n_estimators is not None else profile.get("n_estimators", 5)),
        "--sample-size",
        str(args.sample_size),
        "--test-size",
        str(args.test_size),
        "--random-state",
        str(FIXED_RANDOM_STATE),
        "--compute-backend",
        str(args.compute_backend),
        "--torch-device",
        str(args.torch_device),
        "--out",
        str(out_csv),
    ]

    # Tree params: only pass if CLI explicitly overrides or profile defines them.
    # When omitted, verify_semantic_forest_multi.py resolves from dataset_tree_params.
    _max_depth = args.max_depth if args.max_depth is not None else profile.get("max_depth")
    _min_split = args.min_samples_split if args.min_samples_split is not None else profile.get("min_samples_split")
    _min_leaf = args.min_samples_leaf if args.min_samples_leaf is not None else profile.get("min_samples_leaf")
    if _max_depth is not None:
        cmd.extend(["--max-depth", str(_max_depth)])
    if _min_split is not None:
        cmd.extend(["--min-samples-split", str(_min_split)])
    if _min_leaf is not None:
        cmd.extend(["--min-samples-leaf", str(_min_leaf)])

    if args.datasets:
        cmd.extend(["--datasets", args.datasets])
    if args.all_tasks:
        cmd.append("--all-tasks")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.feature_cache_dir is not None:
        cmd.extend(["--feature-cache-dir", args.feature_cache_dir])

    optional_profile_args = {
        "search_strategy": "--search-strategy",
        "heuristic_probe_samples": "--heuristic-probe-samples",
        "aco_num_ants": "--aco-num-ants",
        "aco_num_iterations": "--aco-num-iterations",
        "aco_alpha": "--aco-alpha",
        "aco_beta": "--aco-beta",
        "aco_evaporation_rate": "--aco-evaporation-rate",
        "aco_q": "--aco-q",
        "aco_explore_prob": "--aco-explore-prob",
    }
    for key_name, cli_name in optional_profile_args.items():
        if key_name in profile:
            cmd.extend([cli_name, str(profile[key_name])])

    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Semantic-Forest-lab algorithm runner"
    )
    parser.add_argument(
        "--algorithm",
        default="id3",
        choices=["id3", "c45", "cart"],
        help="Algorithm profile to run.",
    )
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--sample-size", type=int, default=0, help="Max samples per dataset (0=no limit, use all data)")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--random-state",
        type=int,
        default=FIXED_RANDOM_STATE,
        help="Deprecated: ignored. Random seed is fixed to 42 for all runs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--feature-cache-dir", default=str(Path("output") / "feature_cache"))
    parser.add_argument("--out-dir", default=str(Path("output") / "lab_runs"))
    parser.add_argument("--compute-backend", default="auto", choices=["auto", "numpy", "torch"])
    parser.add_argument("--torch-device", default="auto")

    # Optional overrides (profile defaults if omitted)
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-split", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=None)

    args = parser.parse_args()
    args.random_state = FIXED_RANDOM_STATE

    profile = load_algorithm_profile(args.algorithm)

    print(
        f"[Lab] Algorithm={profile['name']} "
        f"split_criterion={profile['split_criterion']}"
    )
    cmd = build_command(args, profile)
    print(f"[Lab] Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
