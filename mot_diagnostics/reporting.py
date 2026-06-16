from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def dataset_comparison_summary(
    gt_summary: pd.DataFrame,
    det_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge GT difficulty and detection gap into one diagnostic view.
    Higher problem_score => more likely to hurt trackers (esp. HOTA).
    """
    df = gt_summary.copy()
    if det_summary is not None and not det_summary.empty:
        det_cols = [c for c in det_summary.columns if c != "sequence"]
        df = df.merge(det_summary, on="sequence", how="left", suffixes=("", "_det"))

    def col(name: str, default: float = 0.0) -> pd.Series:
        if name in df.columns:
            return df[name].fillna(default)
        return pd.Series(default, index=df.index)

    # Normalized problem indicators (0-1 scale per column)
    indicators = pd.DataFrame({
        "occlusion": _norm(col("occluded_frame_ratio")),
        "crowding": _norm(col("mean_objs_per_frame")),
        "crossing": _norm(col("crossing_events")),
        "motion": _norm(col("mean_track_speed_px")),
        "global_motion": _norm(col("mean_global_motion_px")),
        "short_tracks": _norm(col("short_track_ratio_lt30")),
    })

    if det_summary is not None:
        recall_col = [c for c in df.columns if c.startswith("recall_iou_")]
        if recall_col:
            indicators["low_recall"] = 1.0 - _norm(col(recall_col[0]))
        indicators["fn_occlusion"] = _norm(col("fn_in_occlusion_ratio"))
        indicators["high_fp"] = _norm(1.0 - col("precision_iou_0_5", 1.0))

    df["problem_score"] = indicators.mean(axis=1)
    for c in indicators.columns:
        df[f"indicator_{c}"] = indicators[c]

    rank_cols = [
        "sequence",
        "problem_score",
        "occluded_frame_ratio",
        "mean_objs_per_frame",
        "crossing_events",
        "mean_track_speed_px",
    ]
    if det_summary is not None:
        rank_cols.extend(
            [c for c in df.columns if c.startswith("recall_iou_") or c.startswith("precision_iou_")]
        )

    extra = [c for c in df.columns if c.startswith("indicator_")]
    ordered = rank_cols + extra
    ordered = list(dict.fromkeys(ordered))
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest].sort_values("problem_score", ascending=False)


def _norm(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def plot_sequence_bars(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_path: Path,
    top_n: int = 20,
) -> None:
    if metric not in df.columns or df.empty:
        return

    plot_df = df.nlargest(min(top_n, len(df)), metric)
    fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.35)))
    ax.barh(plot_df["sequence"], plot_df[metric], color="steelblue")
    ax.set_xlabel(metric)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_recall_vs_occlusion(
    frame_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Scatter: per-frame recall vs occlusion level — key DanceTrack diagnostic."""
    if frame_df.empty or "recall" not in frame_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = frame_df["occluded_pairs"].to_numpy()
    y = frame_df["recall"].to_numpy()
    ax.scatter(x, y, alpha=0.25, s=12, c="crimson")
    if len(x) > 5:
        bins = np.arange(0, max(x) + 2)
        bin_recall = []
        bin_centers = []
        for b in range(int(max(x)) + 1):
            mask = x == b
            if mask.sum() > 0:
                bin_recall.append(y[mask].mean())
                bin_centers.append(b)
        ax.plot(bin_centers, bin_recall, "k-o", linewidth=2, label="mean recall")
        ax.legend()

    ax.set_xlabel("GT occluded pairs in frame")
    ax.set_ylabel(f"Detection recall (IoU)")
    ax.set_title("Recall degrades with occlusion?")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_text_report(
    comparison: pd.DataFrame,
    dataset_name: str,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    lines = [
        f"MOT Dataset Diagnostic Report — {dataset_name}",
        "=" * 60,
        "",
    ]

    if comparison.empty:
        lines.append("No data analyzed.")
    else:
        top = comparison.iloc[0]
        lines.extend([
            "Hardest sequences (by composite problem_score):",
            "",
        ])
        for _, row in comparison.head(10).iterrows():
            lines.append(
                f"  {row['sequence']}: score={row.get('problem_score', 0):.3f} "
                f"| occlusion={row.get('occluded_frame_ratio', 0):.2%} "
                f"| objs/frame={row.get('mean_objs_per_frame', 0):.1f}"
            )

        recall_cols = [c for c in comparison.columns if c.startswith("recall_iou_")]
        if recall_cols:
            rc = recall_cols[0]
            mean_recall = comparison[rc].mean()
            lines.extend([
                "",
                f"Mean detection recall ({rc}): {mean_recall:.2%}",
            ])
            worst = comparison.nsmallest(3, rc)
            lines.append("Lowest-recall sequences:")
            for _, row in worst.iterrows():
                lines.append(f"  {row['sequence']}: {row[rc]:.2%}")

        lines.extend([
            "",
            "Interpretation guide:",
            "  - High occlusion + low recall  => detector bottleneck (fix det or use stronger assoc)",
            "  - High crossing + good recall    => association / ReID bottleneck (OC-SORT family)",
            "  - High global_motion             => camera motion hurts motion-based trackers",
            "  - High short_track_ratio         => fragmented GT or fast enter/exit — hurts HOTA",
            "  - fn_in_occlusion_ratio >> fn_not => missing boxes under overlap (common in DanceTrack)",
            "",
        ])

    out_path.write_text("\n".join(lines), encoding="utf-8")
