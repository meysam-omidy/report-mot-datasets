#!/usr/bin/env python3
"""
Full diagnosis: GT difficulty + detection gap + problem ranking.

Usage:
  python scripts/run_full_diagnosis.py --config config.yaml

Run on multiple datasets by editing config.yaml (or pass --config per dataset).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from mot_diagnostics.analyzers.dataset_stats import (  # noqa: E402
    analyze_gt_sequence,
    build_frame_level_gt_report,
)
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
from mot_diagnostics.reporting import (  # noqa: E402
    dataset_comparison_summary,
    plot_recall_vs_occlusion,
    plot_sequence_bars,
    save_csv,
    write_text_report,
)


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
    out_dir = cfg.output.root / cfg.dataset.name / "full_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(cfg.dataset.root, cfg.dataset.sequences)
    if not sequences:
        raise SystemExit(f"No sequences found under {cfg.dataset.root}")

    print(f"=== Full diagnosis: {cfg.dataset.name} ===")
    print(f"GT:  {cfg.dataset.root}")
    print(f"Det: {cfg.detections.root}")

    gt_summary_rows = []
    det_summary_rows = []
    gt_frame_rows = []
    det_frame_rows = []
    size_rows = []

    for seq in sequences:
        gt = read_mot_txt(
            gt_path(cfg.dataset.root, seq),
            ignore_conf_zero=cfg.dataset.ignore_conf_zero,
        )
        info = read_seqinfo(seqinfo_path(cfg.dataset.root, seq))
        gt_summary_rows.append(analyze_gt_sequence(gt, info, cfg.analysis))
        gt_frame_rows.append(build_frame_level_gt_report(gt, info, cfg.analysis))

        det_file = detection_path(
            cfg.detections.root,
            seq,
            layout=cfg.detections.layout,
            extension=cfg.detections.extension,
        )
        if det_file.exists():
            det = read_detections(
                det_file,
                fmt=cfg.detections.format,
                min_conf=cfg.detections.score_threshold,
            )
            det_summary_rows.append(analyze_detection_gap(gt, det, info, cfg.analysis))
            det_frame_rows.append(
                build_frame_level_detection_report(gt, det, info, cfg.analysis)
            )
            size_rows.append(build_size_bucket_report(gt, det, info, cfg.analysis))

    gt_summary = pd.DataFrame(gt_summary_rows)
    det_summary = pd.DataFrame(det_summary_rows) if det_summary_rows else None

    comparison = dataset_comparison_summary(gt_summary, det_summary)
    save_csv(gt_summary, out_dir / "gt_sequence_summary.csv")
    save_csv(comparison, out_dir / "problem_ranking.csv")
    save_csv(pd.concat(gt_frame_rows, ignore_index=True), out_dir / "gt_per_frame.csv")

    if det_summary is not None:
        save_csv(det_summary, out_dir / "detection_sequence_summary.csv")
        save_csv(
            pd.concat(det_frame_rows, ignore_index=True),
            out_dir / "detection_per_frame.csv",
        )
        if size_rows:
            save_csv(pd.concat(size_rows, ignore_index=True), out_dir / "recall_by_size.csv")

        plot_recall_vs_occlusion(
            pd.concat(det_frame_rows, ignore_index=True),
            out_dir / "plots" / "recall_vs_occlusion.png",
        )

    plot_sequence_bars(
        comparison,
        "problem_score",
        f"{cfg.dataset.name}: composite difficulty score",
        out_dir / "plots" / "problem_score.png",
    )

    write_text_report(comparison, cfg.dataset.name, out_dir / "REPORT.txt")

    # Dataset-level aggregates for cross-dataset comparison (run separately per dataset)
    agg = _aggregate_dataset_metrics(gt_summary, det_summary)
    save_csv(pd.DataFrame([agg]), out_dir / "dataset_aggregate.csv")

    print(f"\nTop problem sequences:")
    for _, row in comparison.head(5).iterrows():
        print(f"  {row['sequence']}: problem_score={row['problem_score']:.3f}")
    print(f"\nFull report: {out_dir / 'REPORT.txt'}")


def _aggregate_dataset_metrics(
    gt_summary: pd.DataFrame,
    det_summary: pd.DataFrame | None,
) -> dict:
    agg = {
        "num_sequences": len(gt_summary),
        "mean_occluded_frame_ratio": gt_summary["occluded_frame_ratio"].mean(),
        "mean_objs_per_frame": gt_summary["mean_objs_per_frame"].mean(),
        "mean_crossing_events": gt_summary["crossing_events"].mean(),
        "mean_track_speed_px": gt_summary["mean_track_speed_px"].mean(),
        "mean_global_motion_px": gt_summary["mean_global_motion_px"].mean(),
        "mean_short_track_ratio": gt_summary["short_track_ratio_lt30"].mean(),
    }
    if det_summary is not None and not det_summary.empty:
        recall_cols = [c for c in det_summary.columns if c.startswith("recall_iou_")]
        prec_cols = [c for c in det_summary.columns if c.startswith("precision_iou_")]
        if recall_cols:
            agg["mean_recall"] = det_summary[recall_cols[0]].mean()
        if prec_cols:
            agg["mean_precision"] = det_summary[prec_cols[0]].mean()
        if "fn_in_occlusion_ratio" in det_summary.columns:
            agg["mean_fn_in_occlusion_ratio"] = det_summary["fn_in_occlusion_ratio"].mean()
    return agg


if __name__ == "__main__":
    main()
