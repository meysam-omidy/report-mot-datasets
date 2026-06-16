from __future__ import annotations

from pathlib import Path

from mot_diagnostics.config import AppConfig
from mot_diagnostics.io import discover_sequences

DEFAULT_SPLITS = ("train", "val", "test")
LEGACY_SPLIT = "all"


def discover_splits(dataset_root: Path, configured: list[str] | None = None) -> list[str]:
    """
    Resolve which splits to run.

    - If dataset.root contains train/val/test subfolders -> use those (or configured subset).
    - If dataset.root directly holds sequences (gt/gt.txt per folder) -> single legacy split "all".
    """
    if configured:
        return list(configured)

    found = [
        name
        for name in DEFAULT_SPLITS
        if (dataset_root / name).is_dir() and discover_sequences(dataset_root / name)
    ]
    if found:
        return found

    if discover_sequences(dataset_root):
        return [LEGACY_SPLIT]

    return []


def split_dataset_path(dataset_root: Path, split: str) -> Path:
    if split == LEGACY_SPLIT:
        return dataset_root
    return dataset_root / split


def split_detection_path(det_root: Path, split: str) -> Path:
    """Use det_root/<split> when that folder exists, else flat det_root."""
    if split == LEGACY_SPLIT:
        return det_root
    candidate = det_root / split
    if candidate.is_dir():
        return candidate
    return det_root


def split_output_path(output_root: Path, dataset_name: str, split: str, task: str) -> Path:
    return output_root / dataset_name / split / task


def resolve_split_paths(cfg: AppConfig) -> list[tuple[str, Path, Path]]:
    splits = discover_splits(cfg.dataset.root, cfg.dataset.splits)
    if not splits:
        raise FileNotFoundError(
            f"No splits found under {cfg.dataset.root}. "
            f"Expected subfolders like train/ and val/, or sequence folders directly."
        )

    resolved: list[tuple[str, Path, Path]] = []
    for split in splits:
        ds_path = split_dataset_path(cfg.dataset.root, split)
        if not ds_path.exists():
            raise FileNotFoundError(f"Split folder not found: {ds_path}")
        det_path = split_detection_path(cfg.detections.root, split)
        resolved.append((split, ds_path, det_path))
    return resolved
