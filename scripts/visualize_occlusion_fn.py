#!/usr/bin/env python3
"""
Visualize occlusion vs FN diagnosis on dataset frames.

Uses the same GT-overlap occlusion rule and IoU matching as the detection-gap
analyzer. Draws:
  - green: matched GT (TP)
  - orange: FN with overlapping GT (occluded)
  - red: FN without occlusion (detector miss, not explained by overlap)
  - yellow outline: occluded GT that was still detected
  - blue: unmatched detections (FP)

Usage:
  python scripts/visualize_occlusion_fn.py --config config.yaml --sequence MOT20-01
  python scripts/visualize_occlusion_fn.py --config config.yaml --sequence MOT20-01 --frame 120
  python scripts/visualize_occlusion_fn.py --config config.yaml --sequence MOT20-01 --video
  python scripts/visualize_occlusion_fn.py --config config.yaml --sequence MOT20-01 --hide tp fp
  python scripts/visualize_occlusion_fn.py --config config.yaml --sequence MOT20-01 --show fn_not_occluded fn_occluded

Box visibility is controlled in config.yaml -> visualization (show_tp, show_fp, ...)
or via --show / --hide on the command line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mot_diagnostics.config import VisualizationConfig, load_config, validate_paths  # noqa: E402
from mot_diagnostics.io import (  # noqa: E402
    detection_path,
    discover_sequences,
    gt_path,
    read_detections,
    read_mot_txt,
    read_seqinfo,
    seqinfo_path,
    group_by_frame,
)
from mot_diagnostics.splits import resolve_split_paths  # noqa: E402
from mot_diagnostics.visualize import (  # noqa: E402
    GtCategory,
    diagnose_frame,
    frame_diagnosis_summary,
    format_gt_label,
    pick_frame,
    resolve_frame_image,
    should_draw_gt,
)


COLORS = {
    "tp": (0, 200, 0),
    "fn_occluded": (0, 140, 255),
    "fn_not_occluded": (0, 0, 255),
    "occluded_tp_outline": (0, 255, 255),
    "fp": (255, 120, 0),
    "text_bg": (0, 0, 0),
    "legend_bg": (30, 30, 30),
}


def _xywh_to_xyxy(box: np.ndarray) -> tuple[int, int, int, int]:
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[0] + box[2])
    y2 = int(box[1] + box[3])
    return x1, y1, x2, y2


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y_text = max(th + 4, y)
    cv2.rectangle(img, (x, y_text - th - 4), (x + tw + 4, y_text + baseline), COLORS["text_bg"], -1)
    cv2.putText(img, text, (x + 2, y_text), font, scale, color, thickness, cv2.LINE_AA)


def render_frame(
    image: np.ndarray,
    diag,
    *,
    viz: VisualizationConfig,
    occlusion_iou_threshold: float,
    iou_threshold: float,
    sequence: str,
) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]

    if viz.show_fp:
        for det_idx in diag.fp_det_idx:
            box = diag.det_boxes[det_idx]
            x1, y1, x2, y2 = _xywh_to_xyxy(box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), COLORS["fp"], 2)
            if viz.show_labels:
                _draw_label(canvas, "FP", x1, y1, COLORS["fp"])

    for label in diag.gt_labels:
        if not should_draw_gt(label, viz):
            continue

        x1, y1, x2, y2 = _xywh_to_xyxy(label.box_xywh)
        if label.category == GtCategory.TP:
            color = COLORS["tp"]
        elif label.category == GtCategory.FN_OCCLUDED:
            color = COLORS["fn_occluded"]
        else:
            color = COLORS["fn_not_occluded"]
        tag = format_gt_label(label)

        thickness = 3 if label.category != GtCategory.TP else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        if (
            viz.show_occluded_tp_outline
            and label.category == GtCategory.TP
            and label.is_occluded
        ):
            cv2.rectangle(
                canvas,
                (x1 - 2, y1 - 2),
                (x2 + 2, y2 + 2),
                COLORS["occluded_tp_outline"],
                2,
            )
        if viz.show_labels:
            _draw_label(canvas, tag, x1, max(0, y1 - 6), color)

    if viz.show_legend:
        counts = frame_diagnosis_summary(diag)
        header = (
            f"{sequence} | frame {diag.frame} | "
            f"TP={counts['tp']} FN_occ={counts['fn_occluded']} "
            f"FN_not_occ={counts['fn_not_occluded']} FP={counts['fp']}"
        )
        cv2.rectangle(canvas, (0, 0), (w, 52), COLORS["legend_bg"], -1)
        cv2.putText(
            canvas, header, (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA,
        )

        legend_parts = []
        if viz.show_tp:
            legend_parts.append("Green=TP")
        if viz.show_fn_occluded:
            legend_parts.append("Orange=FN occluded")
        if viz.show_fn_not_occluded:
            legend_parts.append("Red=FN not occluded")
        if viz.show_fp:
            legend_parts.append("Blue=FP")
        legend_line = "  ".join(legend_parts) if legend_parts else "(no categories enabled)"
        cv2.putText(
            canvas, legend_line, (8, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Occlusion: GT IoU>={occlusion_iou_threshold:.2f}  Match IoU>={iou_threshold:.2f}",
            (8, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA,
        )

    return canvas


def load_sequence_data(cfg, split: str, sequence: str):
    dataset_root = None
    detection_root = None
    for s, ds, det in resolve_split_paths(cfg):
        if s == split:
            dataset_root = ds
            detection_root = det
            break
    if dataset_root is None:
        raise ValueError(f"Split {split!r} not found in config")

    seq_dir = dataset_root / sequence
    if not (seq_dir / "gt" / "gt.txt").exists():
        available = discover_sequences(dataset_root, cfg.dataset.sequences)
        raise FileNotFoundError(
            f"Sequence {sequence!r} not found under {dataset_root}. "
            f"Available: {', '.join(available[:10])}{'...' if len(available) > 10 else ''}"
        )

    gt = read_mot_txt(
        gt_path(dataset_root, sequence),
        ignore_conf_zero=cfg.dataset.ignore_conf_zero,
    )
    det_file = detection_path(
        detection_root,
        sequence,
        layout=cfg.detections.layout,
        extension=cfg.detections.extension,
    )
    if not det_file.exists():
        raise FileNotFoundError(f"Detection file not found: {det_file}")

    det = read_detections(
        det_file,
        fmt=cfg.detections.format,
        min_conf=cfg.detections.score_threshold,
    )
    info = read_seqinfo(seqinfo_path(dataset_root, sequence))
    return seq_dir, gt, det, info


def _load_or_blank_frame(
    seq_dir: Path,
    frame: int,
    info,
) -> tuple[np.ndarray, Path | None, bool]:
    img_path = resolve_frame_image(seq_dir, frame)
    if img_path is not None:
        image = cv2.imread(str(img_path))
        if image is not None:
            return image, img_path, False

    if info.im_width and info.im_height:
        blank = np.full((info.im_height, info.im_width, 3), 32, dtype=np.uint8)
        note = (
            f"No img1/ frame image for frame {frame} — "
            f"rendering boxes on {info.im_width}x{info.im_height} blank canvas"
        )
        cv2.putText(
            blank,
            note,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return blank, None, True

    raise FileNotFoundError(
        f"No image found for frame {frame} under {seq_dir / 'img1'} "
        f"and seqinfo.ini has no imWidth/imHeight for a fallback canvas"
    )


def save_still_image(
    cfg,
    seq_dir: Path,
    sequence: str,
    gt,
    det,
    info,
    frame: int,
    out_path: Path,
    viz: VisualizationConfig,
) -> dict:
    gt_by_frame = group_by_frame(gt)
    det_by_frame = group_by_frame(det)
    gdf = gt_by_frame.get(frame, gt.iloc[0:0])
    ddf = det_by_frame.get(frame, det.iloc[0:0])

    image, img_path, used_blank = _load_or_blank_frame(seq_dir, frame, info)

    diag = diagnose_frame(gdf, ddf, frame, cfg.analysis)
    rendered = render_frame(
        image,
        diag,
        viz=viz,
        occlusion_iou_threshold=cfg.analysis.occlusion_iou_threshold,
        iou_threshold=cfg.analysis.iou_thresholds[0],
        sequence=sequence,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), rendered)
    summary = frame_diagnosis_summary(diag)
    summary["frame"] = frame
    summary["image"] = str(img_path) if img_path else ""
    summary["blank_canvas"] = used_blank
    summary["output"] = str(out_path)
    return summary


def save_video(
    cfg,
    seq_dir: Path,
    sequence: str,
    gt,
    det,
    info,
    out_path: Path,
    viz: VisualizationConfig,
    *,
    start_frame: int | None,
    end_frame: int | None,
    fps: float | None,
) -> Path:
    gt_by_frame = group_by_frame(gt)
    det_by_frame = group_by_frame(det)
    frames = sorted(set(gt_by_frame.keys()) | set(det_by_frame.keys()))
    if start_frame is not None:
        frames = [f for f in frames if f >= start_frame]
    if end_frame is not None:
        frames = [f for f in frames if f <= end_frame]
    if not frames:
        raise ValueError("No frames to render for video")

    probe, _, _ = _load_or_blank_frame(seq_dir, frames[0], info)
    h, w = probe.shape[:2]
    video_fps = fps or info.frame_rate or 20.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    written = 0
    for frame in frames:
        try:
            image, _, _ = _load_or_blank_frame(seq_dir, frame, info)
        except FileNotFoundError:
            continue

        gdf = gt_by_frame.get(frame, gt.iloc[0:0])
        ddf = det_by_frame.get(frame, det.iloc[0:0])
        diag = diagnose_frame(gdf, ddf, frame, cfg.analysis)
        rendered = render_frame(
            image,
            diag,
            viz=viz,
            occlusion_iou_threshold=cfg.analysis.occlusion_iou_threshold,
            iou_threshold=cfg.analysis.iou_thresholds[0],
            sequence=sequence,
        )
        writer.write(rendered)
        written += 1

    writer.release()
    if written == 0:
        raise RuntimeError("Video export wrote 0 frames (missing images?)")
    return out_path


VIZ_CATEGORIES = ("tp", "fp", "fn_occluded", "fn_not_occluded", "occluded_tp_outline", "labels", "legend")


def _apply_viz_overrides(viz: VisualizationConfig, args) -> VisualizationConfig:
    """CLI --show / --hide override config.yaml visualization section."""
    viz = VisualizationConfig(
        show_tp=viz.show_tp,
        show_fp=viz.show_fp,
        show_fn_occluded=viz.show_fn_occluded,
        show_fn_not_occluded=viz.show_fn_not_occluded,
        show_occluded_tp_outline=viz.show_occluded_tp_outline,
        show_labels=viz.show_labels,
        show_legend=viz.show_legend,
    )
    field_map = {
        "tp": "show_tp",
        "fp": "show_fp",
        "fn_occluded": "show_fn_occluded",
        "fn_not_occluded": "show_fn_not_occluded",
        "occluded_tp_outline": "show_occluded_tp_outline",
        "labels": "show_labels",
        "legend": "show_legend",
    }
    for name in args.show or []:
        if name not in field_map:
            raise ValueError(f"Unknown category {name!r}. Choose from: {', '.join(VIZ_CATEGORIES)}")
        setattr(viz, field_map[name], True)
    for name in args.hide or []:
        if name not in field_map:
            raise ValueError(f"Unknown category {name!r}. Choose from: {', '.join(VIZ_CATEGORIES)}")
        setattr(viz, field_map[name], False)
    return viz


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize occluded GT vs FN-not-occluded on sample frames or video"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (train/val/test)")
    parser.add_argument("--sequence", type=str, required=True, help="Sequence folder name")
    parser.add_argument("--frame", type=int, default=None, help="Frame index (1-based MOT convention)")
    parser.add_argument(
        "--pick-frame",
        type=str,
        default="worst_fn_not_occluded",
        choices=["worst_fn", "worst_fn_not_occluded", "interesting", "first"],
        help="Auto-pick frame when --frame is omitted",
    )
    parser.add_argument("--video", action="store_true", help="Export annotated MP4 for the sequence")
    parser.add_argument("--video-start", type=int, default=None)
    parser.add_argument("--video-end", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None, help="Video FPS (default: seqinfo frameRate or 20)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder (default: outputs/<dataset>/<split>/visualizations/<sequence>)",
    )
    parser.add_argument(
        "--show",
        nargs="+",
        choices=VIZ_CATEGORIES,
        default=None,
        metavar="CATEGORY",
        help="Force categories on (overrides config): tp fp fn_occluded fn_not_occluded "
        "occluded_tp_outline labels legend",
    )
    parser.add_argument(
        "--hide",
        nargs="+",
        choices=VIZ_CATEGORIES,
        default=None,
        metavar="CATEGORY",
        help="Force categories off (overrides config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    validate_paths(cfg)
    viz = _apply_viz_overrides(cfg.visualization, args)

    seq_dir, gt, det, info = load_sequence_data(cfg, args.split, args.sequence)

    out_root = args.output_dir
    if out_root is None:
        out_root = (
            cfg.output.root
            / cfg.dataset.name
            / args.split
            / "visualizations"
            / args.sequence
        )

    if args.video:
        video_path = out_root / f"{args.sequence}_occlusion_fn.mp4"
        path = save_video(
            cfg,
            seq_dir,
            args.sequence,
            gt,
            det,
            info,
            video_path,
            viz,
            start_frame=args.video_start,
            end_frame=args.video_end,
            fps=args.fps,
        )
        print(f"Video saved: {path}")
        return

    frame = args.frame
    if frame is None:
        frame = pick_frame(gt, det, cfg.analysis, strategy=args.pick_frame)
        print(f"Auto-picked frame {frame} (strategy={args.pick_frame})")

    still_path = out_root / f"{args.sequence}_frame{frame:06d}_occlusion_fn.jpg"
    summary = save_still_image(
        cfg, seq_dir, args.sequence, gt, det, info, frame, still_path, viz
    )

    print(f"Saved: {summary['output']}")
    if summary["blank_canvas"]:
        print("Note: frame images missing — drew boxes on blank canvas from seqinfo.ini")
    else:
        print(f"Source image: {summary['image']}")
    print(
        f"Frame {summary['frame']}: "
        f"TP={summary['tp']} | FN occluded={summary['fn_occluded']} | "
        f"FN not occluded={summary['fn_not_occluded']} | "
        f"occluded GT total={summary['occluded_gt']} | FP={summary['fp']}"
    )


if __name__ == "__main__":
    main()
