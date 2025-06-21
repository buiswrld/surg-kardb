# inspect_data.py
import os, pickle, glob, textwrap
from collections import Counter
import torch

# ------------------ EDIT THESE IF NEEDED -----------------------------------
FRAMES_DIR  = "./joint_out"   # folder with frame_*.pkl
DATASET_PKL = "sandbox/action_dataset_joints_leg_sampled_5.pkl"
# ---------------------------------------------------------------------------

def banner(msg):
    print("\n" + "-"*len(msg))
    print(msg)
    print("-"*len(msg))

import torch
import pickle
import io

def cpu_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except RuntimeError as e:
        # Fallback: use a custom CPU unpickler for nested CUDA tensors
        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)
        with open(path, 'rb') as f:
            return CPU_Unpickler(f).load()

# 1) Locate frame files ------------------------------------------------------
paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.pkl")))
banner(f"Found {len(paths)} frame_*.pkl files")
print("first 3 paths:", paths[:3])

if not paths:
    raise SystemExit("No frame files found – check FRAMES_DIR.")

# 2) Inspect the very first frame ------------------------------------------
first_frame = cpu_load(paths[0])
banner("Keys in first frame")
print(list(first_frame.keys()))

print("\ntrackers (first frame):", first_frame.get("trackers", [])[:10])
j3d = first_frame.get("joints3d")
if j3d is not None:
    print("joints3d shape of first tracker:", j3d[0].shape if hasattr(j3d[0], "shape") else type(j3d[0]))

# 3) Collect unique tracker IDs across all frames (may take a few seconds) --
unique_trackers = []
for p in paths:
    fr = cpu_load(p)
    unique_trackers.extend([str(t) for t in fr.get("trackers", [])])
unique_trackers = list(dict.fromkeys(unique_trackers))  # preserve order

banner("First 10 unique tracker IDs seen in frames")
print(unique_trackers[:10])

# 4) Load dataset identifiers ----------------------------------------------
from sandbox.edge import convert_pkl_to_matrices
ds_samples = convert_pkl_to_matrices(
    pkl_path=DATASET_PKL,
    spatial_pairs=[],
    seq_len=5,
    num_joints=28,
    coords_per_joint=3,
    split="train"
)
dataset_ids = [s["id"] for s in ds_samples]
banner("First 10 dataset IDs from main pickle")
print(dataset_ids[:10])

# 5) Mapping coverage check -------------------------------------------------
suffixes = {id_.split("_")[-1] for id_ in dataset_ids}
missing  = [tid for tid in unique_trackers if tid not in suffixes]

banner("Tracker-ID → Dataset-ID coverage")
print(f"total tracker IDs seen: {len(unique_trackers)}")
print(f"tracker IDs that match a dataset ID suffix: {len(unique_trackers) - len(missing)}")
print(f"tracker IDs with NO matching dataset ID: {len(missing)}")
if missing:
    print("example missing IDs:", missing[:10])

# 6) Quick relaxed collision count -----------------------------------------
try:
    from empirical.collide import detect_collisions
    from empirical.util import read_pickle
    sample_frames = [cpu_load(p) for p in paths[:300]]
    coll_events, _, _ = detect_collisions(sample_frames, radius=2.0, velocity_threshold=0.0)
    banner("Super-relaxed collision test on first 300 frames")
    print("collision events detected:", len(coll_events))
except Exception as e:
    banner("Collision test skipped (import or runtime error)")
    print(e)

    

from empirical.attn import process_files, count_focused_attention_events

d, sdict, total = process_files(FRAMES_DIR)

for margin in (15, 30, 45, 60):
    for window in (10, 5, 3):
        counts, _ = count_focused_attention_events(
            d, margin_of_error=margin, time_frame=window,
            start_frame_dict=sdict, total_num_frames=total
        )
        flat_total = sum(v if isinstance(v, int) else sum(v.values()) for v in counts.values())
        print(f"margin={margin:2}°, window={window:2} ⇒ total events: {flat_total}")


