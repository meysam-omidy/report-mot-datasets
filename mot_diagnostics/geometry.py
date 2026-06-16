from __future__ import annotations

import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    out = np.empty_like(boxes, dtype=np.float64)
    out[:, 0] = boxes[:, 0]
    out[:, 1] = boxes[:, 1]
    out[:, 2] = boxes[:, 0] + boxes[:, 2]
    out[:, 3] = boxes[:, 1] + boxes[:, 3]
    return out


def box_area(boxes_xywh: np.ndarray) -> np.ndarray:
    return np.maximum(boxes_xywh[:, 2], 0) * np.maximum(boxes_xywh[:, 3], 0)


def box_centers(boxes_xywh: np.ndarray) -> np.ndarray:
    cx = boxes_xywh[:, 0] + boxes_xywh[:, 2] * 0.5
    cy = boxes_xywh[:, 1] + boxes_xywh[:, 3] * 0.5
    return np.column_stack([cx, cy])


def iou_matrix(boxes_a_xywh: np.ndarray, boxes_b_xywh: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of boxes."""
    if boxes_a_xywh.size == 0 or boxes_b_xywh.size == 0:
        return np.zeros((len(boxes_a_xywh), len(boxes_b_xywh)), dtype=np.float64)

    a = xywh_to_xyxy(boxes_a_xywh)
    b = xywh_to_xyxy(boxes_b_xywh)

    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    inter_x1 = np.maximum(a[:, 0][:, None], b[:, 0][None, :])
    inter_y1 = np.maximum(a[:, 1][:, None], b[:, 1][None, :])
    inter_x2 = np.minimum(a[:, 2][:, None], b[:, 2][None, :])
    inter_y2 = np.minimum(a[:, 3][:, None], b[:, 3][None, :])

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter = inter_w * inter_h

    union = a_area[:, None] + b_area[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def center_distance_matrix(
    boxes_a_xywh: np.ndarray, boxes_b_xywh: np.ndarray
) -> np.ndarray:
    ca = box_centers(boxes_a_xywh)
    cb = box_centers(boxes_b_xywh)
    diff = ca[:, None, :] - cb[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def greedy_match(
    iou: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Greedy IoU matching (det rows x gt cols).
    Returns (matched_det_idx, matched_gt_idx, matched_iou).
    """
    if iou.size == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )

    pairs: list[tuple[int, int, float]] = []
    for d in range(iou.shape[0]):
        for g in range(iou.shape[1]):
            if iou[d, g] >= iou_threshold:
                pairs.append((d, g, float(iou[d, g])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    used_d: set[int] = set()
    used_g: set[int] = set()
    md, mg, mi = [], [], []
    for d, g, v in pairs:
        if d in used_d or g in used_g:
            continue
        used_d.add(d)
        used_g.add(g)
        md.append(d)
        mg.append(g)
        mi.append(v)

    return (
        np.array(md, dtype=np.int64),
        np.array(mg, dtype=np.int64),
        np.array(mi, dtype=np.float64),
    )
