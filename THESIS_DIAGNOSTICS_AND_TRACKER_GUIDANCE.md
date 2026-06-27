# Thesis Reference: Dataset Diagnostics & OC-SORT Enhancement Strategy

This document consolidates diagnostic findings from the **analysis** project, empirical tracker results from the **Observation-Centric-SORT** reimplementation, and training/integration notes from the **motion-predictor** project. Use it as a source when writing the thesis introduction, motivation, methodology, and discussion sections.

**Related outputs:** `analysis/outputs/occluded_0.1/`, `occluded_0.3/`, `occluded_0.5/` (same detections and GT; only `occlusion_iou_threshold` differs).

---

## 1. Project overview

| Project | Role |
|---------|------|
| **analysis** | Quantifies dataset difficulty and detection gaps (train/val splits, per-sequence rankings). |
| **motion-predictor** | Trains LSTM / Transformer models for motion and (optionally) learned Kalman Q/R. |
| **Observation-Centric-SORT-Rethinking-SORT-for-Robust-Multi-Object-Tracking** | OC-SORT reimplementation; integrates learned motion into Kalman predict/update. |

**Empirical tracker results (reported baseline):**

| Method | Relative gain vs base OC-SORT |
|--------|-------------------------------|
| Full deep-learning motion (Transformer/LSTM + Kalman fusion) | ~**+2%** |
| Simple Kalman with **R scaled by detection confidence** each update | ~**+3.5%** |

Motion models were trained on **all four datasets** (MOT17, MOT20, DanceTrack, SportsMOT). The confidence-R baseline outperforming the full DL pipeline is a central motivation for refocusing the thesis contribution.

---

## 2. How occlusion is defined in diagnostics

For each frame, each GT box is marked **occluded** if its maximum pairwise IoU with any **other** GT box on that frame is ≥ `occlusion_iou_threshold`.

- **Occluded frame ratio:** fraction of frames with at least one such overlapping pair.
- **FN in occlusion ratio:** among unmatched GT boxes (false negatives), fraction that were in an occluded state on that frame.

**What does *not* change when the threshold changes:** recall, precision, crossing events, track speed, acceleration, crowding, short-track ratio, and lowest-recall sequence rankings.

**What changes:** occluded frame ratio, FN-in-occlusion ratio, problem-score components tied to occlusion, and (slightly) composite problem ranking.

Implementation: `mot_diagnostics/analyzers/detection_gap.py`, `dataset_stats.py`.

---

## 3. Occlusion threshold sensitivity (0.1 vs 0.3 vs 0.5)

### 3.1 Interpretation of each threshold

| Threshold | Meaning | Risk if used alone |
|-----------|---------|-------------------|
| **0.1** | Near-contact / slight overlap counts | **Overstates** occlusion on DanceTrack |
| **0.3** | Meaningful body overlap | **Recommended primary** threshold for thesis |
| **0.5** | Strong / major overlap only | **Understates** occlusion; many FNs relabeled as non-occlusion |

### 3.2 DanceTrack — aggregate (val split)

| Metric | IoU ≥ 0.1 | IoU ≥ 0.3 | IoU ≥ 0.5 |
|--------|-----------|-----------|-----------|
| Occluded frames (mean) | 91.8% | 72.4% | **40.4%** |
| FN during occlusion (mean) | 95.2% | 79.8% | **54.6%** |
| FN *not* during occlusion (mean) | 4.8% | 20.2% | **45.4%** |
| Mean recall @ IoU 0.5 | 95.7% | 95.7% | 95.7% |
| Mean crossing events / seq | 36.4 | 36.4 | 36.4 |

Train split shows the same monotonic trend (e.g. FN-in-occlusion: 97.6% → 92.0% → 70.5%).

**Sequences with >30% of FNs outside heavy overlap (val):**

| Threshold | Count (of 25 val seqs) |
|-----------|------------------------|
| 0.1 | 0 |
| 0.3 | 6 (`0018`, `0065`, `0063`, `0094`, `0034`, `0073`) |
| 0.5 | 19 |

### 3.3 MOT17 — aggregate (val split)

| Metric | 0.1 | 0.3 | 0.5 |
|--------|-----|-----|-----|
| Occluded frames | 99.1% | 79.6% | **59.4%** |
| FN in occlusion | 91.4% | 77.8% | **62.2%** |

### 3.4 MOT20 — aggregate (val split)

| Metric | 0.1 | 0.3 | 0.5 |
|--------|-----|-----|-----|
| Occluded frames | 100% | 100% | **96.2%** |
| FN in occlusion | 89.4% | 78.8% | **61.0%** |

At 0.1 and 0.3, **every MOT20 frame** has at least one GT pair with IoU ≥ threshold (extreme crowding). At 0.5, frame-level occlusion drops slightly; FN-in-occlusion still falls to ~61%, separating “crowded scene” from “this specific miss was heavily overlapped.”

### 3.5 Ranking stability

**DanceTrack val — top-10 hardest sequences (by problem_score):**

- **0.1 vs 0.3:** 9/10 overlap (`0014` out, `0063` in at 0.3).
- **0.3 vs 0.5:** 10/10 overlap (same core set; order shifts slightly).

**Lowest-recall sequences unchanged at all thresholds:** `dancetrack0018` (85.7%), `dancetrack0073` (86.4%), `dancetrack0041` (90.5%).

**Conclusion for thesis:** Threshold choice greatly affects **absolute occlusion statistics** and **FN attribution**, but **not** the identity of hard sequences or detection-quality conclusions. Report **0.3 as primary**; include **0.1 and 0.5 as sensitivity analysis**.

---

## 4. Dataset characterisation (primary threshold: IoU ≥ 0.3)

### 4.1 DanceTrack (val)

| Signal | Value | Implication |
|--------|-------|-------------|
| Occluded frames | ~72% | Frequent overlap |
| FN in occlusion | ~80% | Most misses coincide with overlap (at 0.3) |
| Mean recall | ~95.7% | Detector is good on average |
| Crossing events | ~36/seq (up to 203 on `0081`) | Association / ID switches matter |
| Track speed + accel | ~7 px/frame, high accel | Non-linear motion; constant-velocity KF is weak |
| Short-track ratio | ~0.3% | GT fragmentation is not the bottleneck |
| Objects / frame | ~9 | Moderate crowding (vs MOT20) |

**Hardest val sequences (problem_score, @ 0.3):** `0026`, `0043`, `0041`, `0094`, `0073`, `0081`, `0090`, `0034`, `0035`, `0063`.

### 4.2 DanceTrack — sequence-level failure modes (@ 0.3 and 0.5)

| Sequence | Failure mode | Notes @ 0.5 |
|----------|--------------|-------------|
| `0018` | **Detector** (medium boxes) | ~0% FN in heavy overlap; recall 86% |
| `0073`, `0043` | Detector + mixed | Many FNs not in heavy overlap |
| `0041`, `0081` | **Crossing / association** | High crossing (112, 203); only ~26–50% FN in heavy overlap |
| `0026`, `0090`, `0014` | Occlusion + overlap | Still >65% FN in occlusion @ 0.5 |
| `0079`, `0030`, `0077` | Relatively easy | High recall, lower problem score |

### 4.3 MOT17 vs MOT20 (brief)

- **MOT17:** Higher objects/frame (~21 val), more crossing than DanceTrack; occlusion still high at 0.3 (~80% frames val).
- **MOT20:** Extreme crowding (88–125 objs/frame), very high crossing; occlusion metrics saturated at 0.1/0.3.

### 4.4 Diagnostic interpretation guide (from reports)

| Pattern | Likely bottleneck |
|---------|-------------------|
| High occlusion + low recall | Detector or need BYTE / stronger low-score recovery |
| High crossing + good recall | Association (OC-SORT family), not detector |
| High global motion | Camera motion hurts motion-based prediction |
| High short-track ratio | Fragmentation / enter-exit (not DanceTrack’s main issue) |
| High `fn_in_occlusion_ratio` | Missing boxes under overlap (DanceTrack signature at loose thresholds) |

---

## 5. Why confidence-scaled R beats full deep-learning motion

### 5.1 What confidence-R does

Standard SORT Kalman in the codebase scales **R at initialization** by score:

```text
R *= exp(2 * (1 - score))
```

The stronger **+3.5%** baseline applies confidence-based (or similar) **R adjustment on every update**: low-confidence detections are down-weighted; high-confidence ones pull the state back. This modulates **trust**, not **location**.

### 5.2 What the current DL pipeline does

1. Predicts a **new bbox** (Transformer/LSTM) and fuses into Kalman state (`kalman_fusion_blend`).
2. Optionally predicts learned **Q** and **R**.
3. Trains on **one-step bbox error** (NLL + CIoU), not tracker-level metrics.

### 5.3 Why DL underperforms conf-R on DanceTrack

| Factor | Effect |
|--------|--------|
| Bbox fusion during **crossings** | Wrong predictions increase **ID switches** (AssA drops) |
| Conf-R helps **all** low-quality dets | Not only heavily overlapped boxes (~45% of FNs at IoU 0.5 are non-occlusion) |
| Training on clean GT vs detector gaps | Model sees smooth trajectories; tracker sees missing/noisy frames |
| `k_last_updates` includes **predicted** frames when lost | Error compounding during occlusion |
| R from network tied to predict step | Weaker link to **matched detection score** at update time |

**Thesis insight:** Gains come from **heteroscedastic filtering** (adaptive trust), not from replacing the motion model with a bbox regressor.

---

## 6. Recommended thesis contribution

### 6.1 Proposed framing

> **Context-aware adaptive Kalman filtering for observation-centric MOT:** a temporal network predicts **process noise Q** and **measurement noise R** from track history, detection confidence, and overlap context—while **linear Kalman motion and OC-SORT association remain unchanged**.

Position the +3.5% confidence-R result as **motivation**, not as the final method.

### 6.2 Architecture: learned Q/R only (primary)

| Component | Base OC-SORT | Conf-R (+3.5%) | Thesis target |
|-----------|--------------|----------------|---------------|
| Motion | Constant-velocity KF | Same | Same KF |
| R at update | Fixed | `f(det_score)` | `R = g_θ(history, score, overlap)` |
| Q at predict | Fixed | Fixed | `Q = h_θ(history, context)` |
| Network bbox output | — | — | **None** (`kalman_fusion_blend = 0`) |

**Network inputs:** normalized xywh history, velocity/acceleration (13-D features), detection scores, frames since last real observation; optionally max neighbor overlap.

**Network outputs:** `log_var_q` (4), `log_var_r` (4) — mapped to diagonal Q/R in Kalman space (see `kalman_filter.py`: `learned_process_noise_matrix`, `learned_measurement_noise_matrix`).

### 6.3 Optional extension: gated residual

- Apply a **small** learned Δz to KF prediction only when `age > 0` or track is lost.
- **Disable** during high-crossing context (`0081`, `0041`).
- Keep as ablation, not the main claim.

---

## 7. Training recommendations (motion-predictor)

### 7.1 Data

- Train on **detector trajectories** aligned to GT (not GT alone) where possible (`generate_combined_data.ipynb`).
- Include all four datasets; ** overweight or fine-tune on DanceTrack** for target benchmark.
- Match inference context length to tracker (`update_window_end` ≈ 30–50 frames; training `seq_in_len` should align).

### 7.2 Augmentation

| Augmentation | Purpose |
|--------------|---------|
| `noise_prob` / `noise_coeff` (size-proportional) | Realistic detector jitter |
| `random_drop_prob` 0.25–0.4 | Simulated missed frames / occlusion gaps |
| `random_jump` (optional) | Non-linear dance choreography |

### 7.3 Loss

- **Primary:** filtering NLL on innovations (heteroscedastic Gaussian on bbox dimensions).
- **Supervise R** toward squared error between noisy detections and GT on history; increase `r_supervise_coeff` vs current 0.1.
- **Baseline prior:** `R ∝ exp(α(1 - score))`; network learns **residual** on top of confidence-R.
- **Secondary:** small CIoU only if using optional residual head — not for Q/R-only model.

### 7.4 Validation

- Do **not** rely on val bbox loss alone.
- Periodically run OC-SORT on DanceTrack val subset with Q/R-only mode; track HOTA, AssA, DetA, IDSW.

---

## 8. Tracker integration recommendations (OC-SORT)

### 8.1 Critical integration fixes

1. **`kalman_fusion_blend = 0`** for Q/R-only experiments — never overwrite `kf.x[:4]` with network bbox.
2. **Motion history:** feed **real observations only** into the network during `StateLost`; do not include predicted boxes in `k_last_updates` for feature computation (avoids compounding error).
3. **R at update:** combine learned R with **current matched detection score**, not only predict-step outputs.
4. **Align config:** `update_window_start/end` (tracker) must match training sequence length; fix invalid `update_window` keys in ablation configs.

### 8.2 DanceTrack-specific OC-SORT parameters

| Parameter | Suggestion | Rationale |
|-----------|------------|-----------|
| `use_byte` | **On** | Recovers low-score dets under overlap |
| `max_age` | 40–60 | Longer occlusions before track death |
| `delta_t` | 3 | Keep ORU / observation-centric recovery |
| `reupdate_type` | `'constant'` (~0.8) | Virtual updates across gaps |
| `association_speed_direction_coefficient` | **0–0.1** | Dancers change direction quickly |
| `kalman_fusion_blend` | **0** (Q/R mode) | Avoid bbox replacement |

### 8.3 Context-dependent gating (overlap-aware)

Use continuous max pairwise overlap (not only binary threshold):

| Overlap context | Behaviour |
|-----------------|-----------|
| IoU < 0.3 with neighbors | conf-R only; minimal Q inflation |
| 0.3 ≤ IoU < 0.5 | Moderate learned R; small Q during gaps |
| IoU ≥ 0.5 or `age > 0` | Full learned Q/R; KF coasts through gap |
| High crossing (`0081`, `0041`) | No bbox fusion; uncertainty-only; careful association |

---

## 9. Evaluation protocol for thesis experiments

### 9.1 Baseline ladder

1. Base OC-SORT (motion off, BYTE on/off)
2. Confidence-R on every update (~+3.5%)
3. Learned R only
4. Learned Q + R
5. (Optional) Q/R + gated residual

### 9.2 Metrics

- **Global:** HOTA, AssA, DetA, IDF1, IDSW, MOTA.
- **Stratified by sequence type** (from diagnostics):

| Bucket | Example sequences | Expected winner |
|--------|-------------------|-----------------|
| Heavy-overlap FN | `0026`, `0090`, `0073`, `0014` | Learned Q/R > conf-R |
| Crossing-dominant | `0081`, `0041`, `0019` | Association tuning; limit motion correction |
| Non-occlusion FN | `0018`, `0043` @ 0.5 | Detector / BYTE; little from overlap-Q/R |
| Easy | `0079`, `0030`, `0077` | All methods similar |

### 9.3 Fair comparison notes

- Same detection files for all methods (`ocsort_x_dance` on DanceTrack).
- Report occlusion sensitivity table (Section 3) when motivating overlap-aware R.
- If learned Q/R beats conf-R only on occlusion-heavy buckets, that is still a valid thesis result—state it explicitly.

---

## 10. Suggested thesis narrative (one paragraph)

DanceTrack combines frequent dancer overlap, non-linear motion, and challenging crossing events. Diagnostic analysis shows that the share of detection false negatives attributable to occluded ground-truth targets depends strongly on overlap strictness (approximately **95% / 80% / 55%** of FNs for GT pairwise IoU thresholds **0.1 / 0.3 / 0.5** on the validation split), while hardest sequences and recall rankings remain stable. A simple heteroscedastic Kalman filter that scales measurement noise by detection confidence outperforms a full neural bbox predictor integrated into OC-SORT, indicating that **adaptive trust** matters more than **learned displacement** for this benchmark. We therefore propose a learned Q/R module that generalizes confidence-based noise scaling using temporal context and overlap, preserving observation-centric association while improving robustness during partial and full occlusions.

---

## 11. Key figures / tables to generate for thesis

1. **Table:** Occlusion sensitivity (Section 3.2–3.4) for DanceTrack, MOT17, MOT20.
2. **Table:** Baseline ladder (Section 9.1) with HOTA / AssA / DetA / IDSW on DanceTrack val.
3. **Bar or scatter:** Per-sequence problem_score vs recall vs crossing (from `problem_ranking.csv` @ 0.3).
4. **Diagram:** OC-SORT pipeline with learned Q/R injection at predict/update (no bbox fusion).
5. **Stratified results:** Tracker metrics on occlusion-heavy vs crossing-heavy vs easy sequence buckets.

---

## 12. File references

| Resource | Path |
|----------|------|
| Diagnostic outputs (0.1 / 0.3 / 0.5) | `analysis/outputs/occluded_{0.1,0.3,0.5}/` |
| Split aggregates | `{dataset}/_combined/full_diagnosis/split_comparison.csv` |
| Per-sequence rankings | `{dataset}/{split}/full_diagnosis/problem_ranking.csv` |
| Human-readable reports | `{dataset}/{split}/full_diagnosis/REPORT.txt` |
| Occlusion logic (detection FN) | `analysis/mot_diagnostics/analyzers/detection_gap.py` |
| Occlusion logic (GT stats) | `analysis/mot_diagnostics/analyzers/dataset_stats.py` |
| Tracker + Kalman integration | `Observation-Centric-SORT-.../ocsort.py`, `track.py`, `kalman_filter.py` |
| Motion model training | `motion-predictor/learned_noise_motion.py`, `train_improved.py` |
| Learned Q/R inference | `Observation-Centric-SORT-.../motion_predictor.py` |

---

*Document generated for thesis preparation. Primary occlusion threshold for conclusions: **GT pairwise IoU ≥ 0.3**. Sensitivity bounds: **0.1** (loose) and **0.5** (strict).*
