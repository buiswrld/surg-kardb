import os, glob, pickle, numpy as np
from typing import List, Dict
from empirical.util    import read_pickle
from empirical.tool    import detect_engagement_event
from empirical.attn    import compute_gaze_vector
from empirical.collide import calculate_velocity

# ---------- CONFIG ----------
FRAMES_DIR   = "./joint_out"
SAVE_DIR     = "./metrics"
FPS          = 30
ABLATE_SEC   = [1, 2, 3, 4]

# tool‑use thresholds (mentor‑specified)
WRIST_TH = 0.30   # m
ELBOW_TH = 0.60   # m

# gaze‑switch params (tuned)
GAZE_COS        = 0.342   # cos 70° – only big head turns
GAZE_SMOOTH_WIN = 25      # stronger moving-average
SWITCH_STRIDE   = 6       # test every 6th frame

PEL_IDX = 0  # pelvis joint index
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Load frames ----------
paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.pkl")))
if not paths:
    raise RuntimeError("No frame_*.pkl files found in joint_out/")
frames: List[Dict] = [read_pickle(p) for p in paths]
print(f"Loaded {len(frames)} frames from joint_out/")

# ---------- Unit detection ----------
pel_sample = np.array([fr["joints3d"][0][PEL_IDX] for fr in frames[:100]])
unit_scale = 0.001 if np.max(np.abs(pel_sample)) > 10 else 1.0
if unit_scale != 1.0:
    print("⚠ Detected large joint magnitudes; assuming millimetres → metres re‑scale 0.001")

# ---------- Motion (shared) ----------

total_dist, speeds = 0.0, []
prev = None
for fr in frames:
    pos = fr["joints3d"][0][PEL_IDX] * unit_scale
    if prev is not None:
        v = calculate_velocity(prev, pos)
        total_dist += v
        speeds.append(v)
    prev = pos
speed_mean = float(np.mean(speeds)) if speeds else 0.0
speed_std  = float(np.std(speeds))  if speeds else 0.0

# ---------- Gaze vectors (cached & smoothed) ----------
gaze_raw = np.array([compute_gaze_vector(fr["joints3d"][0])[1] for fr in frames])
if GAZE_SMOOTH_WIN > 1:
    kernel = np.ones(GAZE_SMOOTH_WIN) / GAZE_SMOOTH_WIN
    gaze_vecs = np.empty_like(gaze_raw)
    for i in range(3):
        gaze_vecs[:, i] = np.convolve(gaze_raw[:, i], kernel, mode="same")
    gaze_vecs /= np.linalg.norm(gaze_vecs, axis=1, keepdims=True) + 1e-8
else:
    gaze_vecs = gaze_raw

gaze_mean_global = gaze_vecs.mean(0)
gaze_mean_global /= np.linalg.norm(gaze_mean_global) + 1e-8

# ------------- Sanity print -------------
print("Pelvis coord range (m):", np.min(pel_sample*unit_scale), "→", np.max(pel_sample*unit_scale))
print("Total distance (m)   :", total_dist)
print("Mean speed  (m/s)    :", speed_mean * FPS)
print("Std  speed  (m/s)    :", speed_std  * FPS)

# ---------- Ablation loop ----------
for secs in ABLATE_SEC:
    win_frames = secs * FPS

    # Tool engagement
    _d, eng_cnt, _rec = detect_engagement_event(
        time_slice=frames,
        wrist_threshold=WRIST_TH,
        elbow_threshold=ELBOW_TH,
        event_time_threshold=win_frames,
    )


    # Attention changes with smoothed gaze and stride
    attn_changes = sum(
        np.dot(gaze_vecs[i], gaze_vecs[i - win_frames]) < GAZE_COS
        for i in range(win_frames, len(gaze_vecs), SWITCH_STRIDE)
    )
    gaze_stab = float(np.mean(gaze_vecs @ gaze_mean_global))

    metrics = {
        "engagement_events":   int(eng_cnt),
        "total_distance":      float(total_dist),
        "speed_mean":          speed_mean,
        "speed_std":           speed_std,
        "mean_gaze_vector":    gaze_mean_global.astype(np.float32),
        "gaze_stability":      gaze_stab,
        "attention_changes":   int(attn_changes),
        "window_seconds":      secs,
    }

    out_path = os.path.join(SAVE_DIR, f"per_clip_metrics_{secs}s.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(metrics, f)

    print(f"✓ wrote {out_path}")
    for k, v in metrics.items():
        if k != "mean_gaze_vector":
            print(f"   {k:18s}: {v}")
