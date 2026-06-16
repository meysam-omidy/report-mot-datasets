"""Generate tiny synthetic MOT data and run a smoke test."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def write_mot(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        seq = "testseq01"
        gt_dir = tmp_path / "data" / seq
        det_dir = tmp_path / "det"

        # Two tracks crossing with partial overlap on frame 3
        gt_rows = [
            "1,1,100,100,50,80,1,-1,-1,-1",
            "1,2,300,100,50,80,1,-1,-1,-1",
            "2,1,120,105,50,80,1,-1,-1,-1",
            "2,2,280,105,50,80,1,-1,-1,-1",
            "3,1,150,110,50,80,1,-1,-1,-1",
            "3,2,170,110,50,80,1,-1,-1,-1",  # overlap
            "4,1,180,115,50,80,1,-1,-1,-1",
            "4,2,220,115,50,80,1,-1,-1,-1",
        ]
        det_rows = [
            "1,102,101,150,179,0.9",
            "1,298,99,350,181,0.85",
            "2,118,104,169,183,0.88",
            "2,282,106,331,187,0.87",
            "3,152,112,200,190,0.8",
            # missed det in occlusion frame for second object
            "4,179,114,229,194,0.82",
            "4,221,116,270,195,0.79",
        ]

        write_mot(gt_dir / "gt" / "gt.txt", gt_rows)
        write_mot(det_dir / f"{seq}.txt", det_rows)
        (gt_dir / "seqinfo.ini").write_text(
            "[Sequence]\nname=testseq01\nseqLength=4\nimWidth=640\nimHeight=480\nframeRate=20\n",
            encoding="utf-8",
        )

        cfg = {
            "dataset": {
                "name": "smoke_test",
                "root": str(tmp_path / "data"),
                "sequences": [],
                "ignore_conf_zero": True,
            },
            "detections": {
                "root": str(det_dir),
                "layout": "per_sequence",
                "format": "yolox",
                "score_threshold": 0.1,
                "extension": ".txt",
            },
            "output": {"root": str(tmp_path / "out")},
            "analysis": {
                "iou_thresholds": [0.5, 0.75],
                "small_object_area_ratio": 0.002,
                "large_object_area_ratio": 0.05,
                "occlusion_iou_threshold": 0.1,
                "crossing_distance_px": 30,
                "crossing_min_frames": 3,
                "max_gap_frames_for_fragmentation": 5,
            },
        }
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")

        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_full_diagnosis.py"), "--config", str(cfg_path)],
            check=True,
        )
        report = tmp_path / "out" / "smoke_test" / "full_diagnosis" / "REPORT.txt"
        assert report.exists(), "Report not generated"
        print("Smoke test OK")
        print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
