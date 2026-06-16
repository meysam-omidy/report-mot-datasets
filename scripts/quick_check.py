#!/usr/bin/env python3
"""Quick sanity check on one sequence."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mot_diagnostics.analyzers.detection_gap import analyze_detection_gap
from mot_diagnostics.config import load_config
from mot_diagnostics.io import read_detections, read_mot_txt, read_seqinfo, seqinfo_path

cfg = load_config(ROOT / "config.yaml")
seq = "dancetrack0001"
gt = read_mot_txt(cfg.dataset.root / seq / "gt" / "gt.txt", ignore_conf_zero=True)
det = read_detections(
    cfg.detections.root / f"{seq}.txt",
    fmt=cfg.detections.format,
    min_conf=cfg.detections.score_threshold,
)
info = read_seqinfo(seqinfo_path(cfg.dataset.root, seq))
r = analyze_detection_gap(gt, det, info, cfg.analysis)
print(f"Sequence: {seq}")
print(f"GT boxes: {len(gt)}, Det boxes: {len(det)}")
print(f"recall@0.5: {r['recall_iou_0_5']:.2%}")
print(f"precision@0.5: {r['precision_iou_0_5']:.2%}")
