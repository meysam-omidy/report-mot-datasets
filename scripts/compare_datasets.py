#!/usr/bin/env python3
"""
Compare aggregate metrics across multiple dataset configs (e.g. DanceTrack vs MOT17).

Create separate config files:
  config_dancetrack.yaml
  config_mot17.yaml
  config_mot20.yaml

Then run:
  python scripts/compare_datasets.py --configs config_dancetrack.yaml config_mot17.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from mot_diagnostics.config import load_config  # noqa: E402
from mot_diagnostics.reporting import ensure_dir, save_csv  # noqa: E402
from mot_diagnostics.splits import discover_splits, split_output_path  # noqa: E402


def collect_aggregates(cfg) -> list[dict]:
    splits = discover_splits(cfg.dataset.root, cfg.dataset.splits or None)
    rows = []
    for split in splits:
        agg_path = (
            split_output_path(cfg.output.root, cfg.dataset.name, split, "full_diagnosis")
            / "dataset_aggregate.csv"
        )
        if not agg_path.exists():
            print(f"Warning: missing {agg_path}")
            continue
        row = pd.read_csv(agg_path).iloc[0].to_dict()
        row["dataset"] = cfg.dataset.name
        row["split"] = split
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset MOT comparison")
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        required=True,
        help="Config files (one per dataset)",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only aggregate existing full_diagnosis outputs",
    )
    args = parser.parse_args()

    aggregates = []
    for config_path in args.configs:
        cfg = load_config(config_path)

        if not args.skip_run:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_full_diagnosis.py"),
                    "--config",
                    str(config_path),
                ],
                check=True,
            )

        aggregates.extend(collect_aggregates(cfg))

    if not aggregates:
        raise SystemExit("No aggregate results found.")

    df = pd.DataFrame(aggregates)
    out_path = ensure_dir(ROOT / "outputs" / "cross_dataset") / "comparison.csv"
    save_csv(df, out_path)

    print("\nCross-dataset comparison:")
    cols = [
        "dataset",
        "split",
        "mean_occluded_frame_ratio",
        "mean_objs_per_frame",
        "mean_crossing_events",
        "mean_recall",
        "mean_fn_in_occlusion_ratio",
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
