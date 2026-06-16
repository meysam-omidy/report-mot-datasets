from __future__ import annotations

from pathlib import Path

import pandas as pd

from mot_diagnostics.analyzers.dataset_stats import (
    analyze_gt_sequence,
    build_frame_level_gt_report,
)
from mot_diagnostics.analyzers.detection_gap import (
    analyze_detection_gap,
    build_frame_level_detection_report,
    build_size_bucket_report,
)
from mot_diagnostics.config import AppConfig
from mot_diagnostics.io import (
    detection_path,
    discover_sequences,
    gt_path,
    read_detections,
    read_mot_txt,
    read_seqinfo,
    seqinfo_path,
)
from mot_diagnostics.reporting import (
    dataset_comparison_summary,
    plot_recall_vs_occlusion,
    plot_sequence_bars,
    save_csv,
    write_text_report,
)
from mot_diagnostics.splits import split_output_path


def run_dataset_analysis_split(
    cfg: AppConfig,
    split: str,
    dataset_root: Path,
) -> Path:
    out_dir = split_output_path(cfg.output.root, cfg.dataset.name, split, "dataset_only")
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(dataset_root, cfg.dataset.sequences)
    if not sequences:
        raise ValueError(f"No sequences under {dataset_root}")

    summary_rows = []
    frame_rows = []

    for seq in sequences:
        gt = read_mot_txt(
            gt_path(dataset_root, seq),
            ignore_conf_zero=cfg.dataset.ignore_conf_zero,
        )
        info = read_seqinfo(seqinfo_path(dataset_root, seq))
        summary_rows.append(analyze_gt_sequence(gt, info, cfg.analysis))
        frame_rows.append(build_frame_level_gt_report(gt, info, cfg.analysis))
        print(f"    {seq}: {len(gt)} GT boxes, {gt['id'].nunique()} tracks")

    summary = pd.DataFrame(summary_rows)
    frames = pd.concat(frame_rows, ignore_index=True)

    save_csv(summary, out_dir / "gt_sequence_summary.csv")
    save_csv(frames, out_dir / "gt_per_frame.csv")

    plot_sequence_bars(
        summary,
        "occluded_frame_ratio",
        f"{cfg.dataset.name}/{split}: occluded frame ratio",
        out_dir / "plots" / "occlusion_by_sequence.png",
    )
    plot_sequence_bars(
        summary,
        "crossing_events",
        f"{cfg.dataset.name}/{split}: crossing events",
        out_dir / "plots" / "crossing_by_sequence.png",
    )
    plot_sequence_bars(
        summary,
        "mean_objs_per_frame",
        f"{cfg.dataset.name}/{split}: objects per frame",
        out_dir / "plots" / "crowding_by_sequence.png",
    )
    return out_dir


def run_detection_analysis_split(
    cfg: AppConfig,
    split: str,
    dataset_root: Path,
    detection_root: Path,
) -> Path:
    out_dir = split_output_path(cfg.output.root, cfg.dataset.name, split, "detection_gap")
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(dataset_root, cfg.dataset.sequences)
    if not sequences:
        raise ValueError(f"No sequences under {dataset_root}")

    summary_rows = []
    frame_rows = []
    size_rows = []

    for seq in sequences:
        gt = read_mot_txt(
            gt_path(dataset_root, seq),
            ignore_conf_zero=cfg.dataset.ignore_conf_zero,
        )
        det_file = detection_path(
            detection_root,
            seq,
            layout=cfg.detections.layout,
            extension=cfg.detections.extension,
        )
        if not det_file.exists():
            print(f"    SKIP {seq}: detection file missing ({det_file})")
            continue

        det = read_detections(
            det_file,
            fmt=cfg.detections.format,
            min_conf=cfg.detections.score_threshold,
        )
        info = read_seqinfo(seqinfo_path(dataset_root, seq))

        summary_rows.append(analyze_detection_gap(gt, det, info, cfg.analysis))
        frame_rows.append(build_frame_level_detection_report(gt, det, info, cfg.analysis))
        size_rows.append(build_size_bucket_report(gt, det, info, cfg.analysis))

        rec = summary_rows[-1].get("recall_iou_0_5", float("nan"))
        print(f"    {seq}: recall@0.5={rec:.2%} | det={len(det)} gt={len(gt)}")

    if not summary_rows:
        raise ValueError(f"No sequences with detections for split '{split}'")

    summary = pd.DataFrame(summary_rows)
    frames = pd.concat(frame_rows, ignore_index=True)
    sizes = pd.concat(size_rows, ignore_index=True) if size_rows else pd.DataFrame()

    save_csv(summary, out_dir / "detection_sequence_summary.csv")
    save_csv(frames, out_dir / "detection_per_frame.csv")
    if not sizes.empty:
        save_csv(sizes, out_dir / "detection_recall_by_size.csv")

    plot_recall_vs_occlusion(frames, out_dir / "plots" / "recall_vs_occlusion.png")
    return out_dir


def run_full_diagnosis_split(
    cfg: AppConfig,
    split: str,
    dataset_root: Path,
    detection_root: Path,
) -> Path:
    out_dir = split_output_path(cfg.output.root, cfg.dataset.name, split, "full_diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(dataset_root, cfg.dataset.sequences)
    if not sequences:
        raise ValueError(f"No sequences under {dataset_root}")

    gt_summary_rows = []
    det_summary_rows = []
    gt_frame_rows = []
    det_frame_rows = []
    size_rows = []

    for seq in sequences:
        gt = read_mot_txt(
            gt_path(dataset_root, seq),
            ignore_conf_zero=cfg.dataset.ignore_conf_zero,
        )
        info = read_seqinfo(seqinfo_path(dataset_root, seq))
        gt_summary_rows.append(analyze_gt_sequence(gt, info, cfg.analysis))
        gt_frame_rows.append(build_frame_level_gt_report(gt, info, cfg.analysis))

        det_file = detection_path(
            detection_root,
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
        f"{cfg.dataset.name}/{split}: composite difficulty score",
        out_dir / "plots" / "problem_score.png",
    )

    write_text_report(
        comparison,
        f"{cfg.dataset.name} ({split})",
        out_dir / "REPORT.txt",
    )

    agg = aggregate_dataset_metrics(gt_summary, det_summary)
    agg["split"] = split
    save_csv(pd.DataFrame([agg]), out_dir / "dataset_aggregate.csv")
    return out_dir


def aggregate_dataset_metrics(
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
