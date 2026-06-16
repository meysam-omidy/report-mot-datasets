#!/usr/bin/env python3
"""
Compare YOLOX detections against GT.

Runs on each split under dataset.root (train, val, ...) automatically.

Usage:
  python scripts/run_detection_analysis.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mot_diagnostics.config import load_config, validate_paths  # noqa: E402
from mot_diagnostics.runner import run_detection_analysis_split  # noqa: E402
from mot_diagnostics.splits import resolve_split_paths  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="MOT detection vs GT diagnostics")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_paths(cfg)

    print(f"Dataset: {cfg.dataset.name}")
    print(f"Root: {cfg.dataset.root}")
    print(f"Detections: {cfg.detections.root} (format={cfg.detections.format})")

    for split, dataset_root, detection_root in resolve_split_paths(cfg):
        print(f"\n=== Split: {split} ===")
        print(f"  GT:  {dataset_root}")
        print(f"  Det: {detection_root}")
        out_dir = run_detection_analysis_split(cfg, split, dataset_root, detection_root)
        print(f"  Saved: {out_dir}")


if __name__ == "__main__":
    main()
