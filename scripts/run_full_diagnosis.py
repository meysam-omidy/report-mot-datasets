#!/usr/bin/env python3
"""
Full diagnosis: GT difficulty + detection gap + problem ranking.

Runs on each split under dataset.root (train, val, ...) automatically.
Results: outputs/<dataset_name>/<split>/full_diagnosis/

Usage:
  python scripts/run_full_diagnosis.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from mot_diagnostics.config import load_config, validate_paths  # noqa: E402
from mot_diagnostics.runner import run_full_diagnosis_split  # noqa: E402
from mot_diagnostics.reporting import save_csv  # noqa: E402
from mot_diagnostics.splits import resolve_split_paths, split_output_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Full MOT dataset + detection diagnosis")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_paths(cfg)

    print(f"=== Full diagnosis: {cfg.dataset.name} ===")
    print(f"Root: {cfg.dataset.root}")
    print(f"Detections: {cfg.detections.root}")

    split_aggs = []

    for split, dataset_root, detection_root in resolve_split_paths(cfg):
        print(f"\n--- Split: {split} ---")
        print(f"  GT:  {dataset_root}")
        print(f"  Det: {detection_root}")
        out_dir = run_full_diagnosis_split(cfg, split, dataset_root, detection_root)

        agg_path = out_dir / "dataset_aggregate.csv"
        if agg_path.exists():
            split_aggs.append(pd.read_csv(agg_path).iloc[0].to_dict())

        ranking = pd.read_csv(out_dir / "problem_ranking.csv")
        print("  Top problem sequences:")
        for _, row in ranking.head(3).iterrows():
            print(f"    {row['sequence']}: problem_score={row['problem_score']:.3f}")
        print(f"  Report: {out_dir / 'REPORT.txt'}")

    if len(split_aggs) > 1:
        combined = pd.DataFrame(split_aggs)
        combined_out = split_output_path(
            cfg.output.root, cfg.dataset.name, "_combined", "full_diagnosis"
        )
        combined_out.mkdir(parents=True, exist_ok=True)
        save_csv(combined, combined_out / "split_comparison.csv")
        print(f"\nSplit comparison: {combined_out / 'split_comparison.csv'}")


if __name__ == "__main__":
    main()
