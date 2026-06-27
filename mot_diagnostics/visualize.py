from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from mot_diagnostics.config import AnalysisConfig, VisualizationConfig
from mot_diagnostics.geometry import greedy_match, iou_matrix
from mot_diagnostics.io import df_to_boxes, group_by_frame


class GtCategory(str, Enum):
    TP = "tp"
    FN_OCCLUDED = "fn_occluded"
    FN_NOT_OCCLUDED = "fn_not_occluded"


@dataclass
class GtBoxLabel:
    track_id: int
    box_xywh: np.ndarray
    category: GtCategory
    is_occluded: bool
    max_occlusion_iou: float


@dataclass
class FrameDiagnosis:
    frame: int
    gt_labels: list[GtBoxLabel]
    det_boxes: np.ndarray
    matched_det_idx: np.ndarray
    fp_det_idx: np.ndarray


def occlusion_mask(gt_boxes: np.ndarray, occlusion_iou_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (occluded_mask, max_iou_per_gt) using the same rule as detection_gap."""
    n = len(gt_boxes)
    mask = np.zeros(n, dtype=bool)
    max_iou = np.zeros(n, dtype=np.float64)
    if n < 2:
        return mask, max_iou

    giou = iou_matrix(gt_boxes, gt_boxes)
    np.fill_diagonal(giou, 0.0)
    max_iou = giou.max(axis=1)
    mask = max_iou >= occlusion_iou_threshold
    return mask, max_iou


def diagnose_frame(
    gdf: pd.DataFrame,
    ddf: pd.DataFrame,
    frame: int,
    cfg: AnalysisConfig,
) -> FrameDiagnosis:
    gt_boxes = df_to_boxes(gdf)
    det_boxes = df_to_boxes(ddf)
    primary_thr = cfg.iou_thresholds[0]

    occluded_mask, max_iou = occlusion_mask(gt_boxes, cfg.occlusion_iou_threshold)

    iou = iou_matrix(det_boxes, gt_boxes)
    md, mg, _ = greedy_match(iou, primary_thr)
    matched_gt = set(mg.tolist())
    matched_det = set(md.tolist())

    gt_labels: list[GtBoxLabel] = []
    for g_idx in range(len(gt_boxes)):
        track_id = int(gdf.iloc[g_idx]["id"]) if not gdf.empty else -1
        if g_idx in matched_gt:
            category = GtCategory.TP
        elif occluded_mask[g_idx]:
            category = GtCategory.FN_OCCLUDED
        else:
            category = GtCategory.FN_NOT_OCCLUDED

        gt_labels.append(
            GtBoxLabel(
                track_id=track_id,
                box_xywh=gt_boxes[g_idx],
                category=category,
                is_occluded=bool(occluded_mask[g_idx]),
                max_occlusion_iou=float(max_iou[g_idx]),
            )
        )

    fp_det_idx = np.array(
        [d for d in range(len(det_boxes)) if d not in matched_det],
        dtype=np.int64,
    )

    return FrameDiagnosis(
        frame=frame,
        gt_labels=gt_labels,
        det_boxes=det_boxes,
        matched_det_idx=md,
        fp_det_idx=fp_det_idx,
    )


def pick_frame(
    gt: pd.DataFrame,
    det: pd.DataFrame,
    cfg: AnalysisConfig,
    strategy: str = "worst_fn_not_occluded",
) -> int:
    """Choose a representative frame for visualization."""
    gt_by_frame = group_by_frame(gt)
    det_by_frame = group_by_frame(det)
    frames = sorted(set(gt_by_frame.keys()) | set(det_by_frame.keys()))
    if not frames:
        raise ValueError("No frames found in GT or detections")

    best_frame = frames[0]
    best_score = -1

    for frame in frames:
        gdf = gt_by_frame.get(frame, pd.DataFrame())
        ddf = det_by_frame.get(frame, pd.DataFrame())
        diag = diagnose_frame(gdf, ddf, frame, cfg)

        fn_occ = sum(1 for g in diag.gt_labels if g.category == GtCategory.FN_OCCLUDED)
        fn_not = sum(1 for g in diag.gt_labels if g.category == GtCategory.FN_NOT_OCCLUDED)
        fn_total = fn_occ + fn_not
        occluded = sum(1 for g in diag.gt_labels if g.is_occluded)

        if strategy == "worst_fn":
            score = fn_total
        elif strategy == "worst_fn_not_occluded":
            score = fn_not
        elif strategy == "interesting":
            score = fn_not * 2 + fn_occ + occluded
        elif strategy == "first":
            return frame
        else:
            score = fn_not

        if score > best_score:
            best_score = score
            best_frame = frame

    return best_frame


def resolve_frame_image(seq_dir: Path, frame: int) -> Path | None:
    """Locate a frame image under standard MOT-style layouts."""
    for img_subdir in ("img1", "img"):
        img_dir = seq_dir / img_subdir
        if not img_dir.is_dir():
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            for pad in (6, 8, 5, 4):
                candidate = img_dir / f"{frame:0{pad}d}{ext}"
                if candidate.exists():
                    return candidate

    for img_subdir in ("img1", "img"):
        img_dir = seq_dir / img_subdir
        if not img_dir.is_dir():
            continue
        matches = sorted(img_dir.glob(f"*{frame}*"))
        if matches:
            return matches[0]
    return None


def should_draw_gt(label: GtBoxLabel, viz: VisualizationConfig) -> bool:
    if label.category == GtCategory.TP:
        return viz.show_tp
    if label.category == GtCategory.FN_OCCLUDED:
        return viz.show_fn_occluded
    return viz.show_fn_not_occluded


def format_gt_label(label: GtBoxLabel) -> str:
    iou_suffix = f" i={label.max_occlusion_iou:.2f}" if label.is_occluded else ""
    if label.category == GtCategory.TP:
        tag = f"id{label.track_id}"
        if label.is_occluded:
            tag += f" occ+det{iou_suffix}"
        return tag
    if label.category == GtCategory.FN_OCCLUDED:
        return f"id{label.track_id} FN occ{iou_suffix}"
    return f"id{label.track_id} FN"


def frame_diagnosis_summary(diag: FrameDiagnosis) -> dict[str, int]:
    counts = {
        "tp": 0,
        "fn_occluded": 0,
        "fn_not_occluded": 0,
        "occluded_gt": 0,
        "fp": len(diag.fp_det_idx),
    }
    for g in diag.gt_labels:
        if g.is_occluded:
            counts["occluded_gt"] += 1
        counts[g.category.value] += 1
    return counts
