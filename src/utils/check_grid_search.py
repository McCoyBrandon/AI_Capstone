"""
check_grid_search.py

Scan a grid-search results folder (e.g., runs/grid_run_202512) and find the run
with the best value of a user-specified metric inside each metrics.json.

Examples:
    python -m src.utils.check_grid_search --metric macro_f1 --root runs/grid_run_202512
    python -m src.utils.check_grid_search --metric weighted_f1 --root runs/grid_run_202512
    
Metric options:
> Primary
accuracy
macro_f1
weighted_f1
balanced_accuracy
> AUROC scores Per Class
NoFailure_auc
TWF_auc
HDF_auc
PWF_auc
OSF_auc
> Aggregate AUROC scores
macro_auc
micro_auc
weighted_auc
binary_auc
"""

import os
import json
import argparse


def find_best_run(root_dir: str, metric_name: str):
    """
    Search through all run subdirectories inside root_dir looking for metrics.json
    and pick the run with the highest value for metric_name.

    Returns:
        (best_path, best_run_name, best_score) or (None, None, None)
    """
    if not os.path.isdir(root_dir):
        print(f"[ERROR] Directory does not exist: {root_dir}")
        return None, None, None

    best_score = None
    best_path = None
    best_run_name = None

    for entry in sorted(os.listdir(root_dir)):
        run_dir = os.path.join(root_dir, entry)
        if not os.path.isdir(run_dir):
            continue

        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.isfile(metrics_path):
            continue

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not parse {metrics_path}: {e}")
            continue

        if metric_name not in metrics:
            print(f"[WARN] Metric '{metric_name}' not found in {metrics_path}, skipping.")
            continue

        score = metrics[metric_name]

        if best_score is None or score > best_score:
            best_score = score
            best_path = run_dir
            best_run_name = entry

    return best_path, best_run_name, best_score


def main():
    parser = argparse.ArgumentParser(description="Find best run based on a metric from grid search metrics.json files.")
    parser.add_argument(
        "--root",
        "-r",
        type=str,
        default=os.path.join("runs", "Test_Run_1"),
        help="Grid search run root directory (default: runs/Test_Run_1)."
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default="macro_f1",
        help="Metric key to maximize from metrics.json (default: macro_f1)."
    )

    args = parser.parse_args()

    print(f"\nScanning folder: {args.root}")
    print(f"Using metric: {args.metric}")

    best_path, best_name, best_score = find_best_run(args.root, args.metric)

    if best_path is None:
        print("\nNo compatible runs found.")
        return

    print("\nBest Run Found:")
    print(f"Run Name : {best_name}")
    print(f"Location : {os.path.abspath(best_path)}")
    print(f"{args.metric}: {best_score:.6f}")


if __name__ == "__main__":
    main()