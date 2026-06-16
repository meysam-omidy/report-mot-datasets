# MOT Dataset & Detection Diagnostics

Toolkit to find **core dataset and detection bottlenecks** for multiple object tracking — especially useful for **DanceTrack** vs **MOT17/MOT20** when tuning OC-SORT or similar trackers.

## What it measures

### Ground truth (dataset) difficulty
| Metric | Why it matters for HOTA |
|--------|-------------------------|
| Occluded frame ratio | Overlapping dancers → ID switches, fragmentations |
| Objects per frame | Crowding stresses association |
| Crossing events | Proximity interactions → OC-SORT weak point |
| Track speed / acceleration | Fast motion breaks linear motion models |
| Global scene motion | Camera movement hurts Kalman / velocity cues |
| Short tracks | Low track length caps HOTA association scores |
| Small / large object ratios | Scale affects detector & matcher |

### Detection gap (YOLOX vs GT)
| Metric | Why it matters |
|--------|----------------|
| Recall / precision @ IoU 0.5, 0.75 | Upper bound on tracker performance |
| FN in occlusion vs not | Is DanceTrack hurt by missing boxes under overlap? |
| Per-frame recall vs occlusion | Pinpoints hard frames |
| Recall by size bucket | Small-dancer misses? |
| Localization & size ratio error | Box noise hurts IoU-based association |

## Setup

```bash
pip install -r requirements.txt
```

## Configure paths

Edit **`config.yaml`** (all paths are variables you set):

```yaml
dataset:
  name: dancetrack
  root: "D:/datasets/DanceTrack/train"   # folder of sequence subfolders

detections:
  root: "D:/detections/yolox_dancetrack_train"
  layout: per_sequence
  format: yolox                 # yolox (frame,x1,y1,x2,y2,conf) | mot (10-field)
  score_threshold: 0.1
```

### Expected layout

**Dataset (MOT standard):**
```
<dataset_root>/
  dancetrack0001/
    gt/gt.txt
    seqinfo.ini          # optional but recommended
    img1/                # not required for analysis
  dancetrack0002/
    ...
```

**Detections (YOLOX output — 6 fields per line):**
```
<det_root>/
  dancetrack0001.txt     # layout: per_sequence
```

YOLOX format (set `detections.format: yolox` in config):
```
frame,x1,y1,x2,y2,conf
```

Standard MOT detection format (set `detections.format: mot`):
```
frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z
```

GT rows with `conf=0` are ignored by default (MOT convention).

## Run

```bash
# GT-only difficulty (no detections needed)
python scripts/run_dataset_analysis.py --config config.yaml

# Detection vs GT gap
python scripts/run_detection_analysis.py --config config.yaml

# Full report + problem ranking + REPORT.txt
python scripts/run_full_diagnosis.py --config config.yaml
```

### Compare DanceTrack vs MOT17 vs MOT20

Copy `config.yaml` to three files, set `dataset.name`, `dataset.root`, and `detections.root` for each, then:

```bash
python scripts/compare_datasets.py --configs config_dancetrack.yaml config_mot17.yaml config_mot20.yaml
```

## Outputs

Under `outputs/<dataset_name>/`:

| Path | Content |
|------|---------|
| `full_diagnosis/problem_ranking.csv` | Sequences ranked by composite difficulty |
| `full_diagnosis/REPORT.txt` | Human-readable summary |
| `full_diagnosis/detection_sequence_summary.csv` | Per-seq recall, FN-in-occlusion, etc. |
| `full_diagnosis/plots/recall_vs_occlusion.png` | Key DanceTrack diagnostic |
| `cross_dataset/comparison.csv` | Side-by-side dataset aggregates |

## Interpreting results for OC-SORT improvements

1. **High `fn_in_occlusion_ratio` + decent recall elsewhere** → invest in detector (or higher-res input), not only association.
2. **Good recall but high `crossing_events` / `occluded_frame_ratio`** → association/ReID/camera-motion modules (your OC-SORT tweaks target this).
3. **High `mean_global_motion_px`** → add camera compensation or rely less on constant-velocity Kalman.
4. **Low recall on `small` size bucket** → tune YOLOX thresholds or multi-scale inference.

Run full diagnosis on DanceTrack first, then compare aggregates with MOT17/MOT20 to see which failure mode is *unique* to DanceTrack.
