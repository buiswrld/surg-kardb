"""
generate_metrics.py  —  minimal-patch version (TRACKER-ID SUFFIX FIX)
Keys now match convert_pkl_to_matrices():  cam_clip_start_tracker
"""

import os, pickle, numpy as np
from collections import defaultdict

# empirical imports ----------------------------------------------------------
from empirical.attn       import process_files as attn_process_files, count_focused_attention_events
from empirical.collide    import detect_collisions
from empirical.group_attn import group_focused_attention
from empirical.group_prox import process_files as prox_process_files, calculate_distance
from empirical.group_dist import get_distance_diff

# dataset import (for correct ID mapping) ------------------------------------
from sandbox.edge import convert_pkl_to_matrices 

# CONFIG ---------------------------------------------------------------------
FRAMES_DIR = "./joint_out"           
DATASET_PKL = "sandbox/action_dataset_joints_leg_sampled_150.pkl"
SAVE_DIR = "./metrics"
SEQ_LEN = 150
os.makedirs(SAVE_DIR, exist_ok=True)

# HELPERS --------------------------------------------------------------------
def _load_frames(folder: str):
    from empirical.util import read_pickle
    paths = sorted(p for p in os.listdir(folder) if p.endswith(".pkl"))
    return [read_pickle(os.path.join(folder, p)) for p in paths]

def _tracker_to_full_id(dataset_pkl: str):
    """
    Build {tracker_id(str) : cam_clip_start_tracker(str)} using the MAIN dataset.
    Handles both '2' and '2.0' forms by mapping both → same full ID.
    """
    mapping = {}
    samples = convert_pkl_to_matrices(
        pkl_path=dataset_pkl,
        spatial_pairs=[],
        seq_len=SEQ_LEN, num_joints=28, coords_per_joint=3,
        split="train"
    )
    for s in samples:
        full_id = s["id"]                       # e.g. c2_20_0_2.0
        float_suffix = full_id.split("_")[-1]    # 2.0
        int_suffix   = str(int(float(float_suffix)))  # 2
        mapping[int_suffix]   = full_id
        mapping[float_suffix] = full_id          # harmless duplicate
    return mapping

# ---------- Tunable params ----------------------------------------
ATTN_MARGIN = 30        # degrees
ATTN_WINDOW = 5         # consecutive frames
COLL_RADIUS = 2.0       # meters
COLL_V_THRESH = 0.0     # m/s (0 = ignore velocity check)
# ---------------------------------------------------------------------------

# INDIVIDUAL PERSON METRICS -------------------------------------------------------------
def compute_attention_events(frames_dir):
    data_dict, start_dict, total_frames = attn_process_files(frames_dir)
    raw, _ = count_focused_attention_events(
        data_dict,
        margin_of_error=ATTN_MARGIN,
        time_frame=ATTN_WINDOW,
        start_frame_dict=start_dict,
        total_num_frames=total_frames,
    )
    flat = {}
    for sub in (raw.values() if isinstance(next(iter(raw.values())), dict) else [raw]):
        for tid, cnt in sub.items():
            flat[str(tid)] = int(cnt)
    return flat

def compute_collision_counts(frames):
    events, _, _ = detect_collisions(frames, radius=COLL_RADIUS, velocity_threshold=COLL_V_THRESH)
    counts = defaultdict(int)
    for ev in events:
        counts[str(ev["person_1"])] += 1
        counts[str(ev["person_2"])] += 1
    return counts


# GROUP METRICS --------------------------------------------------------------
def compute_group_attention(frames, min_frames=15, vec_thresh=2):
    per_frame = [{t: fr["joints3d"][i] for i, t in enumerate(fr["trackers"])} for fr in frames]
    focus, _, _ = group_focused_attention(per_frame, min_frames, vec_thresh)
    return {"__GROUP_ATTENTION_FRAMES__": int(focus)}

def compute_group_proximity(frames_dir):
    data = prox_process_files(frames_dir)
    disp, drift = calculate_distance(data)
    return {"__MEAN_CENTROID_DISPERSION__": float(np.mean(disp)),
            "__TOTAL_CENTROID_DRIFT__":      float(np.sum(drift))}

def compute_group_distribution(frames):
    joints = [fr["joints3d"] for fr in frames]
    trackers = [fr["trackers"] for fr in frames]
    return {"__TOTAL_PAIRWISE_MOTION__": float(sum(get_distance_diff(joints, trackers)))}

# ------------------------------ MAIN ----------------------------------------
if __name__ == "__main__":
    print("➜ loading frames …")
    frames = _load_frames(FRAMES_DIR)
    t2id   = _tracker_to_full_id(DATASET_PKL)     #  <-- FIXED MAPPING

    print("➜ computing metrics …")
    attn_raw = compute_attention_events(FRAMES_DIR)
    coll_raw = compute_collision_counts(frames)

    # re-key to full IDs ------------------------------------------------------
    attention_events = {t2id[k]: v for k, v in attn_raw.items() if k in t2id}
    collision_counts = {t2id[k]: v for k, v in coll_raw.items() if k in t2id}

    # group metrics
    group_attention    = compute_group_attention(frames)
    group_proximity    = compute_group_proximity(FRAMES_DIR)
    group_distribution = compute_group_distribution(frames)

    # SAVE -------------------------------------------------------------------
    with open(os.path.join(SAVE_DIR, "attn_events_per_sample.pkl"), "wb") as f:
        pickle.dump(attention_events, f)
    with open(os.path.join(SAVE_DIR, "collision_count_per_sample.pkl"), "wb") as f:
        pickle.dump(collision_counts, f)

    grp = {}; grp.update(group_attention); grp.update(group_proximity); grp.update(group_distribution)
    with open(os.path.join(SAVE_DIR, "group_level_metrics.pkl"), "wb") as f:
        pickle.dump(grp, f)

    # sanity
    print(f"    ✔ attention_events: {len(attention_events):>4} keys")
    print(f"    ✔ collision_counts: {len(collision_counts):>4} keys")
    print("\nAll metric dictionaries written to:", SAVE_DIR)
