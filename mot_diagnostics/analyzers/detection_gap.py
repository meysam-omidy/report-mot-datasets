from __future__ import annotations

import numpy as np
import pandas as pd

from mot_diagnostics.config import AnalysisConfig
from mot_diagnostics.geometry import box_area, greedy_match, iou_matrix
from mot_diagnostics.io import SeqInfo, df_to_boxes, group_by_frame


def analyze_detection_gap(
    gt: pd.DataFrame,
    det: pd.DataFrame,
    seq_info: SeqInfo,
    cfg: AnalysisConfig,
) -> dict:
    """
    Compare detections against GT — identifies recall/precision bottlenecks.
    """
    gt_by_frame = group_by_frame(gt)
    det_by_frame = group_by_frame(det)
    all_frames = sorted(set(gt_by_frame.keys()) | set(det_by_frame.keys()))

    frame_area = None
    if seq_info.im_width and seq_info.im_height:
        frame_area = seq_info.im_width * seq_info.im_height

    totals = {
        "tp": {t: 0 for t in cfg.iou_thresholds},
        "fp": {t: 0 for t in cfg.iou_thresholds},
        "fn": {t: 0 for t in cfg.iou_thresholds},
    }
    loc_errors: list[float] = []
    size_ratio_errors: list[float] = []
    missed_in_occlusion = 0
    missed_not_occlusion = 0
    fn_total = 0

    conf_matched: list[float] = []
    conf_fp: list[float] = []

    for frame in all_frames:
        gdf = gt_by_frame.get(frame, pd.DataFrame())
        ddf = det_by_frame.get(frame, pd.DataFrame())

        gt_boxes = df_to_boxes(gdf)
        det_boxes = df_to_boxes(ddf)

        # Occlusion context for this frame (from GT)
        occluded_gt_mask = np.zeros(len(gt_boxes), dtype=bool)
        if len(gt_boxes) >= 2:
            giou = iou_matrix(gt_boxes, gt_boxes)
            np.fill_diagonal(giou, 0.0)
            occluded_gt_mask = giou.max(axis=1) >= cfg.occlusion_iou_threshold

        if len(det_boxes) == 0 and len(gt_boxes) == 0:
            continue

        iou = iou_matrix(det_boxes, gt_boxes)

        for thr in cfg.iou_thresholds:
            md, mg, mi = greedy_match(iou, thr)
            totals["tp"][thr] += len(md)
            totals["fp"][thr] += len(det_boxes) - len(md)
            totals["fn"][thr] += len(gt_boxes) - len(mg)

        # Use primary threshold (first in list, typically 0.5) for detailed stats
        primary_thr = cfg.iou_thresholds[0]
        md, mg, mi = greedy_match(iou, primary_thr)

        matched_gt = set(mg.tolist())
        for g_idx in range(len(gt_boxes)):
            if g_idx not in matched_gt:
                fn_total += 1
                if occluded_gt_mask[g_idx]:
                    missed_in_occlusion += 1
                else:
                    missed_not_occlusion += 1

        for d_idx, g_idx, iou_val in zip(md, mg, mi):
            if ddf.empty or gdf.empty:
                continue
            conf_matched.append(float(ddf.iloc[d_idx]["conf"]))

            db = det_boxes[d_idx]
            gb = gt_boxes[g_idx]
            dc = np.array([db[0] + db[2] / 2, db[1] + db[3] / 2])
            gc = np.array([gb[0] + gb[2] / 2, gb[1] + gb[3] / 2])
            loc_errors.append(float(np.linalg.norm(dc - gc)))

            det_a = max(box_area(det_boxes[d_idx : d_idx + 1])[0], 1e-6)
            gt_a = max(box_area(gt_boxes[g_idx : g_idx + 1])[0], 1e-6)
            size_ratio_errors.append(float(det_a / gt_a))

        matched_det = set(md.tolist())
        for d_idx in range(len(det_boxes)):
            if d_idx not in matched_det and not ddf.empty:
                conf_fp.append(float(ddf.iloc[d_idx]["conf"]))

    result = {
        "sequence": seq_info.name,
        "num_frames": len(all_frames),
        "num_gt": len(gt),
        "num_det": len(det),
        "mean_det_per_frame": float(len(det) / max(len(all_frames), 1)),
        "mean_gt_per_frame": float(len(gt) / max(len(all_frames), 1)),
        "det_gt_count_ratio": float(len(det) / max(len(gt), 1)),
        "mean_loc_error_px": float(np.mean(loc_errors)) if loc_errors else np.nan,
        "median_loc_error_px": float(np.median(loc_errors)) if loc_errors else np.nan,
        "mean_size_ratio_det_over_gt": float(np.mean(size_ratio_errors)) if size_ratio_errors else np.nan,
        "fn_in_occlusion_ratio": float(missed_in_occlusion / max(fn_total, 1)),
        "fn_not_in_occlusion_ratio": float(missed_not_occlusion / max(fn_total, 1)),
        "mean_conf_matched": float(np.mean(conf_matched)) if conf_matched else np.nan,
        "mean_conf_fp": float(np.mean(conf_fp)) if conf_fp else np.nan,
    }

    for thr in cfg.iou_thresholds:
        tp = totals["tp"][thr]
        fp = totals["fp"][thr]
        fn = totals["fn"][thr]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        key = str(thr).replace(".", "_")
        result[f"precision_iou_{key}"] = float(prec)
        result[f"recall_iou_{key}"] = float(rec)
        result[f"f1_iou_{key}"] = float(f1)

    return result


def build_frame_level_detection_report(
    gt: pd.DataFrame,
    det: pd.DataFrame,
    seq_info: SeqInfo,
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    """Per-frame detection gap — pinpoints hard frames for trackers."""
    gt_by_frame = group_by_frame(gt)
    det_by_frame = group_by_frame(det)
    all_frames = sorted(set(gt_by_frame.keys()) | set(det_by_frame.keys()))
    primary_thr = cfg.iou_thresholds[0]
    rows = []

    for frame in all_frames:
        gdf = gt_by_frame.get(frame, pd.DataFrame())
        ddf = det_by_frame.get(frame, pd.DataFrame())
        gt_boxes = df_to_boxes(gdf)
        det_boxes = df_to_boxes(ddf)

        occluded_pairs = 0
        if len(gt_boxes) >= 2:
            giou = iou_matrix(gt_boxes, gt_boxes)
            np.fill_diagonal(giou, 0.0)
            occluded_pairs = int(np.sum(giou >= cfg.occlusion_iou_threshold) // 2)

        iou = iou_matrix(det_boxes, gt_boxes)
        md, mg, _ = greedy_match(iou, primary_thr)

        rows.append({
            "sequence": seq_info.name,
            "frame": frame,
            "num_gt": len(gt_boxes),
            "num_det": len(det_boxes),
            "tp": len(md),
            "fp": len(det_boxes) - len(md),
            "fn": len(gt_boxes) - len(mg),
            "recall": len(mg) / max(len(gt_boxes), 1),
            "precision": len(md) / max(len(det_boxes), 1),
            "occluded_pairs": occluded_pairs,
        })

    return pd.DataFrame(rows)


def build_size_bucket_report(
    gt: pd.DataFrame,
    det: pd.DataFrame,
    seq_info: SeqInfo,
    cfg: AnalysisConfig,
) -> pd.DataFrame:
    """Recall breakdown by GT object size — reveals detector weakness."""
    if not (seq_info.im_width and seq_info.im_height):
        return pd.DataFrame()

    frame_area = seq_info.im_width * seq_info.im_height
    gt_by_frame = group_by_frame(gt)
    det_by_frame = group_by_frame(det)
    primary_thr = cfg.iou_thresholds[0]

    buckets = {
        "small": (0.0, cfg.small_object_area_ratio),
        "medium": (cfg.small_object_area_ratio, cfg.large_object_area_ratio),
        "large": (cfg.large_object_area_ratio, 1.0),
    }
    stats = {b: {"tp": 0, "fn": 0} for b in buckets}

    for frame, gdf in gt_by_frame.items():
        ddf = det_by_frame.get(frame, pd.DataFrame())
        gt_boxes = df_to_boxes(gdf)
        det_boxes = df_to_boxes(ddf)
        iou = iou_matrix(det_boxes, gt_boxes)
        _, mg, _ = greedy_match(iou, primary_thr)
        matched_gt = set(mg.tolist())

        for g_idx, box in enumerate(gt_boxes):
            area_ratio = box_area(box.reshape(1, -1))[0] / frame_area
            bucket = "medium"
            for name, (lo, hi) in buckets.items():
                if lo <= area_ratio < hi:
                    bucket = name
                    break
            if area_ratio >= cfg.large_object_area_ratio:
                bucket = "large"

            if g_idx in matched_gt:
                stats[bucket]["tp"] += 1
            else:
                stats[bucket]["fn"] += 1

    rows = []
    for bucket, s in stats.items():
        total = s["tp"] + s["fn"]
        rows.append({
            "sequence": seq_info.name,
            "size_bucket": bucket,
            "tp": s["tp"],
            "fn": s["fn"],
            "recall": s["tp"] / max(total, 1),
            "num_gt": total,
        })
    return pd.DataFrame(rows)
