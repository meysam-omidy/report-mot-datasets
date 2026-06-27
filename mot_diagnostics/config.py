from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    name: str = "dancetrack"
    root: Path = Path(".")
    # Splits under root to process (e.g. train, val). Empty = auto-detect train/val/test.
    splits: list[str] = field(default_factory=list)
    sequences: list[str] = field(default_factory=list)
    ignore_conf_zero: bool = True


@dataclass
class DetectionConfig:
    root: Path = Path(".")
    layout: str = "per_sequence"  # per_sequence | per_sequence_folder | flat
    # mot: frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z
    # yolox: frame,x1,y1,x2,y2,conf  (YOLOX / ByteTrack detection output)
    format: str = "yolox"
    score_threshold: float = 0.1
    extension: str = ".txt"


@dataclass
class OutputConfig:
    root: Path = Path("./outputs")


@dataclass
class AnalysisConfig:
    iou_thresholds: list[float] = field(default_factory=lambda: [0.5, 0.75])
    small_object_area_ratio: float = 0.002
    large_object_area_ratio: float = 0.05
    occlusion_iou_threshold: float = 0.1
    crossing_distance_px: float = 30.0
    crossing_min_frames: int = 3
    max_gap_frames_for_fragmentation: int = 5


@dataclass
class VisualizationConfig:
    """Which box categories to draw in occlusion/FN visualizations."""
    show_tp: bool = True
    show_fp: bool = True
    show_fn_occluded: bool = True
    show_fn_not_occluded: bool = True
    show_occluded_tp_outline: bool = True
    show_labels: bool = True
    show_legend: bool = True


@dataclass
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    detections: DetectionConfig = field(default_factory=DetectionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def _to_path(value: Any) -> Path:
    text = str(value).strip().strip('"').strip("'")
    # Normalize accidental Windows double-slash after drive letter (D://foo -> D:/foo)
    if len(text) >= 3 and text[1] == ":" and text[2] in "/\\":
        text = text[0:2] + text[2:].lstrip("/\\")
        if not text[2:].startswith(("/", "\\")):
            text = text[0:2] + "\\" + text[2:]
    return Path(text).expanduser()


def validate_paths(cfg: AppConfig, *, require_detections: bool = True) -> None:
    """Fail fast with a clear message when configured paths are missing."""
    from mot_diagnostics.splits import discover_splits

    if not cfg.dataset.root.exists():
        hint = _path_hint(cfg.dataset.root)
        raise FileNotFoundError(
            f"Dataset root not found: {cfg.dataset.root}\n"
            f"Edit config.yaml -> dataset.root\n"
            f"{hint}"
        )

    splits = discover_splits(cfg.dataset.root, cfg.dataset.splits or None)
    if not splits:
        raise FileNotFoundError(
            f"No splits found under {cfg.dataset.root}\n"
            f"Expected subfolders train/, val/ (and optionally test/) with sequences inside."
        )

    if require_detections and not cfg.detections.root.exists():
        hint = _path_hint(cfg.detections.root)
        raise FileNotFoundError(
            f"Detections root not found: {cfg.detections.root}\n"
            f"Edit config.yaml -> detections.root\n"
            f"{hint}"
        )


def _path_hint(path: Path) -> str:
    """Suggest a fix when user omits a parent folder (e.g. D:/.Datasets vs D:/Projects/.Datasets)."""
    parts = path.parts
    if len(parts) >= 2 and parts[1].startswith("."):
        candidate = Path(parts[0]) / "Projects" / Path(*parts[1:])
        if candidate.exists():
            return f"Did you mean: {candidate}"
    return "Check that the folder exists and the drive letter is correct."


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    dataset_raw = raw.get("dataset", {})
    det_raw = raw.get("detections", {})
    output_raw = raw.get("output", {})
    analysis_raw = raw.get("analysis", {})
    viz_raw = raw.get("visualization", {})

    cfg = AppConfig(
        dataset=DatasetConfig(
            name=str(dataset_raw.get("name", "dancetrack")),
            root=_to_path(dataset_raw.get("root", ".")),
            splits=list(dataset_raw.get("splits") or []),
            sequences=list(dataset_raw.get("sequences") or []),
            ignore_conf_zero=bool(dataset_raw.get("ignore_conf_zero", True)),
        ),
        detections=DetectionConfig(
            root=_to_path(det_raw.get("root", ".")),
            layout=str(det_raw.get("layout", "per_sequence")),
            format=str(det_raw.get("format", "yolox")),
            score_threshold=float(det_raw.get("score_threshold", 0.1)),
            extension=str(det_raw.get("extension", ".txt")),
        ),
        output=OutputConfig(root=_to_path(output_raw.get("root", "./outputs"))),
        analysis=AnalysisConfig(
            iou_thresholds=list(analysis_raw.get("iou_thresholds", [0.5, 0.75])),
            small_object_area_ratio=float(
                analysis_raw.get("small_object_area_ratio", 0.002)
            ),
            large_object_area_ratio=float(
                analysis_raw.get("large_object_area_ratio", 0.05)
            ),
            occlusion_iou_threshold=float(
                analysis_raw.get("occlusion_iou_threshold", 0.1)
            ),
            crossing_distance_px=float(analysis_raw.get("crossing_distance_px", 30)),
            crossing_min_frames=int(analysis_raw.get("crossing_min_frames", 3)),
            max_gap_frames_for_fragmentation=int(
                analysis_raw.get("max_gap_frames_for_fragmentation", 5)
            ),
        ),
        visualization=VisualizationConfig(
            show_tp=bool(viz_raw.get("show_tp", True)),
            show_fp=bool(viz_raw.get("show_fp", True)),
            show_fn_occluded=bool(viz_raw.get("show_fn_occluded", True)),
            show_fn_not_occluded=bool(viz_raw.get("show_fn_not_occluded", True)),
            show_occluded_tp_outline=bool(viz_raw.get("show_occluded_tp_outline", True)),
            show_labels=bool(viz_raw.get("show_labels", True)),
            show_legend=bool(viz_raw.get("show_legend", True)),
        ),
    )
    return cfg
