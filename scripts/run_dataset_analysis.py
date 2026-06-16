#!/usr/bin/env python3
"""
Analyze GT annotations only — occlusion, crowding, motion, track structure.

Usage:
  python scripts/run_dataset_analysis.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mot_diagnostics.analyzers.dataset_stats import (  # noqa: E402
    analyze_gt_sequence,
    build_frame_level_gt_report,
)
from mot_diagnostics.config import load_config, validate_paths  # noqa: E402
from mot_diagnostics.io import (  # noqa: E402
    discover_sequences,
    gt_path,
    read_mot_txt,
    read_seqinfo,
    seqinfo_path,
)
from mot_diagnostics.reporting import (  # noqa: E402
    plot_sequence_bars,
    save_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MOT GT dataset diagnostics")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="Path to config.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_paths(cfg, require_detections=False)
    out_dir = cfg.output.root / cfg.dataset.name / "dataset_only"
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(cfg.dataset.root, cfg.dataset.sequences)
    if not sequences:
        raise SystemExit(f"No sequences found under {cfg.dataset.root}")

    print(f"Dataset: {cfg.dataset.name}")
    print(f"Root: {cfg.dataset.root}")
    print(f"Sequences: {len(sequences)}")

    summary_rows = []
    frame_rows = []

    for seq in sequences:
        gt = read_mot_txt(
            gt_path(cfg.dataset.root, seq),
            ignore_conf_zero=cfg.dataset.ignore_conf_zero,
        )
        info = read_seqinfo(seqinfo_path(cfg.dataset.root, seq))
        summary_rows.append(analyze_gt_sequence(gt, info, cfg.analysis))
        frame_rows.append(build_frame_level_gt_report(gt, info, cfg.analysis))
        print(f"  {seq}: {len(gt)} GT boxes, {gt['id'].nunique()} tracks")

    summary = __import__("pandas").DataFrame(summary_rows)
    frames = __import__("pandas").concat(frame_rows, ignore_index=True)

    save_csv(summary, out_dir / "gt_sequence_summary.csv")
    save_csv(frames, out_dir / "gt_per_frame.csv")

    plot_sequence_bars(
        summary,
        "occluded_frame_ratio",
        f"{cfg.dataset.name}: occluded frame ratio",
        out_dir / "plots" / "occlusion_by_sequence.png",
    )
    plot_sequence_bars(
        summary,
        "crossing_events",
        f"{cfg.dataset.name}: crossing events",
        out_dir / "plots" / "crossing_by_sequence.png",
    )
    plot_sequence_bars(
        summary,
        "mean_objs_per_frame",
        f"{cfg.dataset.name}: objects per frame",
        out_dir / "plots" / "crowding_by_sequence.png",
    )

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
