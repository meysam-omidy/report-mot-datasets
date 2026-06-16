from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MOT_COLUMNS = [
    "frame",
    "id",
    "bb_left",
    "bb_top",
    "bb_width",
    "bb_height",
    "conf",
    "x",
    "y",
    "z",
]

YOLOX_COLUMNS = ["frame", "x1", "y1", "x2", "y2", "conf"]


@dataclass
class SeqInfo:
    name: str
    seq_length: int | None = None
    im_width: int | None = None
    im_height: int | None = None
    frame_rate: float | None = None


def _finalize_boxes(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with standard bb_left, bb_top, bb_width, bb_height columns."""
    df = df.dropna(subset=["frame", "bb_left", "bb_top", "bb_width", "bb_height"])
    df["frame"] = df["frame"].astype(int)
    df["id"] = df.get("id", pd.Series(-1, index=df.index)).fillna(-1).astype(int)
    df["bb_width"] = df["bb_width"].clip(lower=0)
    df["bb_height"] = df["bb_height"].clip(lower=0)
    return df.reset_index(drop=True)


def read_mot_txt(
    path: Path,
    *,
    ignore_conf_zero: bool = False,
    min_conf: float | None = None,
) -> pd.DataFrame:
    """Load MOT Challenge format file (comma-separated, 10 fields)."""
    if not path.exists():
        raise FileNotFoundError(f"MOT file not found: {path}")

    df = pd.read_csv(path, header=None, names=MOT_COLUMNS, comment="#")
    for col in MOT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _finalize_boxes(df)

    if ignore_conf_zero:
        df = df[df["conf"] != 0]

    if min_conf is not None:
        df = df[df["conf"] >= min_conf]

    return df.reset_index(drop=True)


def read_yolox_txt(
    path: Path,
    *,
    min_conf: float | None = None,
) -> pd.DataFrame:
    """
    Load YOLOX-style detections: frame, x1, y1, x2, y2, conf
    Converts xyxy -> xywh for internal use.
    """
    if not path.exists():
        raise FileNotFoundError(f"Detection file not found: {path}")

    df = pd.read_csv(path, header=None, names=YOLOX_COLUMNS, comment="#")
    for col in YOLOX_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["frame", "x1", "y1", "x2", "y2", "conf"])
    df["frame"] = df["frame"].astype(int)
    df["bb_left"] = df["x1"]
    df["bb_top"] = df["y1"]
    df["bb_width"] = df["x2"] - df["x1"]
    df["bb_height"] = df["y2"] - df["y1"]
    df["id"] = -1

    if min_conf is not None:
        df = df[df["conf"] >= min_conf]

    return _finalize_boxes(df)


def read_detections(
    path: Path,
    *,
    fmt: str = "yolox",
    min_conf: float | None = None,
) -> pd.DataFrame:
    """Load detection file using the configured format."""
    fmt = fmt.lower().replace("-", "_")
    if fmt in ("yolox", "yolox_xyxy", "frame_xyxy_conf"):
        return read_yolox_txt(path, min_conf=min_conf)
    if fmt in ("mot", "motchallenge", "mot_challenge"):
        return read_mot_txt(path, min_conf=min_conf)
    raise ValueError(
        f"Unknown detection format: {fmt!r}. Use 'yolox' or 'mot'."
    )


def read_seqinfo(path: Path) -> SeqInfo:
    name = path.parent.name
    if not path.exists():
        return SeqInfo(name=name)

    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    if "Sequence" not in parser:
        return SeqInfo(name=name)

    sec = parser["Sequence"]
    return SeqInfo(
        name=name,
        seq_length=int(sec.get("seqLength", 0)) or None,
        im_width=int(sec.get("imWidth", 0)) or None,
        im_height=int(sec.get("imHeight", 0)) or None,
        frame_rate=float(sec.get("frameRate", 0)) or None,
    )


def discover_sequences(dataset_root: Path, names: list[str] | None = None) -> list[str]:
    """Find sequence folders that contain gt/gt.txt."""
    if names:
        return sorted(names)

    seqs = []
    for child in sorted(dataset_root.iterdir()):
        if child.is_dir() and (child / "gt" / "gt.txt").exists():
            seqs.append(child.name)
    return seqs


def gt_path(dataset_root: Path, seq_name: str) -> Path:
    return dataset_root / seq_name / "gt" / "gt.txt"


def seqinfo_path(dataset_root: Path, seq_name: str) -> Path:
    return dataset_root / seq_name / "seqinfo.ini"


def detection_path(
    det_root: Path,
    seq_name: str,
    *,
    layout: str,
    extension: str,
) -> Path:
    if layout == "per_sequence":
        return det_root / f"{seq_name}{extension}"
    if layout == "per_sequence_folder":
        return det_root / seq_name / "det.txt"
    if layout == "flat":
        return det_root / f"detections{extension}"
    raise ValueError(f"Unknown detection layout: {layout}")


def group_by_frame(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    if df.empty:
        return {}
    return {int(f): g.reset_index(drop=True) for f, g in df.groupby("frame", sort=True)}


def df_to_boxes(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.zeros((0, 4), dtype=np.float64)
    return df[["bb_left", "bb_top", "bb_width", "bb_height"]].to_numpy(dtype=np.float64)
