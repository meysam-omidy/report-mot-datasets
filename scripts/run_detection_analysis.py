#!/usr/bin/env python3
"""
Compare YOLOX (MOT-format) detections against GT.

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

import pandas as pd  # noqa: E402

from mot_diagnostics.analyzers.detection_gap import (  # noqa: E402
    analyze_detection_gap,
    build_frame_level_detection_report,
    build_size_bucket_report,
)
from mot_diagnostics.config import load_config, validate_paths  # noqa: E402
from mot_diagnostics.io import (  # noqa: E402
    detection_path,
    discover_sequences,
    gt_path,
    read_detections,
    read_mot_txt,
    read_seqinfo,
    seqinfo_path,
)
from mot_diagnostics.reporting import plot_recall_vs_occlusion, save_csv  # noqa: E402


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
    out_dir = cfg.output.root / cfg.dataset.name / "detection_gap"
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(cfg.dataset.root, cfg.dataset.sequences)
    if not sequences:
        raise SystemExit(f"No sequences found under {cfg.dataset.root}")

    print(f"Dataset: {cfg.dataset.name}")
    print(f"GT root: {cfg.dataset.root}")
    print(f"Det root: {cfg.detections.root} (layout={cfg.detections.layout})")
    print(f"Det format: {cfg.detections.format}")

    summary_rows = []
    frame_rows = []
    size_rows = []

    for seq in sequences:
        gt = read_mot_txt(
            gt_path(cfg.dataset.root, seq),
            ignore_conf_zero=cfg.dataset.ignore_conf_zero,
        )
        det_file = detection_path(
            cfg.detections.root,
            seq,
            layout=cfg.detections.layout,
            extension=cfg.detections.extension,
        )
        if not det_file.exists():
            print(f"  SKIP {seq}: detection file missing ({det_file})")
            continue

        det = read_detections(
            det_file,
            fmt=cfg.detections.format,
            min_conf=cfg.detections.score_threshold,
        )
        info = read_seqinfo(seqinfo_path(cfg.dataset.root, seq))

        summary_rows.append(analyze_detection_gap(gt, det, info, cfg.analysis))
        frame_rows.append(build_frame_level_detection_report(gt, det, info, cfg.analysis))
        size_rows.append(build_size_bucket_report(gt, det, info, cfg.analysis))

        rec = summary_rows[-1].get("recall_iou_0_5", float("nan"))
        print(f"  {seq}: recall@0.5={rec:.2%} | det={len(det)} gt={len(gt)}")

    if not summary_rows:
        raise SystemExit("No sequences with both GT and detections were analyzed.")

    summary = pd.DataFrame(summary_rows)
    frames = pd.concat(frame_rows, ignore_index=True)
    sizes = pd.concat(size_rows, ignore_index=True) if size_rows else pd.DataFrame()

    save_csv(summary, out_dir / "detection_sequence_summary.csv")
    save_csv(frames, out_dir / "detection_per_frame.csv")
    if not sizes.empty:
        save_csv(sizes, out_dir / "detection_recall_by_size.csv")

    plot_recall_vs_occlusion(
        frames,
        out_dir / "plots" / "recall_vs_occlusion.png",
    )

    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
