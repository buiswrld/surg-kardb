import os
import sys
import math
import pickle
import statistics

# --------------------------------------------------------------------- CONFIG
METRIC_DIR   = "./metrics"
DATA_PKL     = "sandbox/action_dataset_joints_leg_sampled_5.pkl"

PERSON_METRIC_FILES = (
    "attn_events_per_sample.pkl",
    "collision_count_per_sample.pkl",
)
GROUP_METRIC_FILE   = "group_level_metrics.pkl"

# ---------------------------------------------------------------- UTILITIES
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def quick_stats(values):
    return {
        "min":  min(values),
        "max":  max(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }

def fail(msg):
    print(f"✗ {msg}")
    sys.exit(1)

def ok(msg):
    print(f"✓ {msg}")

# ---------------------------------------------------------------- 1. STRUCTURAL
for fname in (*PERSON_METRIC_FILES, GROUP_METRIC_FILE):
    path = os.path.join(METRIC_DIR, fname)
    if not os.path.isfile(path):
        fail(f"missing {path}")
    obj = load_pickle(path)
    if not isinstance(obj, dict):
        fail(f"{fname} is not a dict")
    for k, v in obj.items():
        if not isinstance(k, str):
            fail(f"{fname}: key {k!r} not str")
        if not isinstance(v, (int, float)):
            fail(f"{fname}: value {v!r} not scalar")
ok("structural check (files present, dict[str → scalar])")

# ---------------------------------------------------------------- 2. KEY-ALIGNMENT
print("Loading dataset identifiers …")
from sandbox.edge import convert_pkl_to_matrices 

samples = convert_pkl_to_matrices(
    pkl_path=DATA_PKL,
    spatial_pairs=[], 
    seq_len=5,
    num_joints=28,
    coords_per_joint=3,
    split="train"
)
dataset_ids = {s["id"] for s in samples}

for fname in PERSON_METRIC_FILES:
    d = load_pickle(os.path.join(METRIC_DIR, fname))
    unknown = [k for k in d if k not in dataset_ids]
    if unknown:
        fail(f"{fname}: unknown IDs (showing 5) {unknown[:5]}")
ok("key-alignment: all person-metric keys exist in dataset")

# ---------------------------------------------------------------- 3. RANGE SANITY
for fname in PERSON_METRIC_FILES:
    vals = load_pickle(os.path.join(METRIC_DIR, fname)).values()
    st   = quick_stats(vals)
    print(f"{fname} stats: {st}")
    if st["max"] > 1e4:
        fail(f"{fname}: max value suspiciously high (>1e4)")
    if math.isnan(st["mean"]):
        fail(f"{fname}: mean is NaN")
ok("range sanity: values look plausible")

# ---------------------------------------------------------------- 4. INTEGRATION WITH GNNDataset
try:
    from sandbox.gcn_dataset import GNNDataset
except ImportError:
    fail("could not import gcn_dataset – adjust PYTHONPATH?")

metric_dicts = {fn: load_pickle(os.path.join(METRIC_DIR, fn))
                for fn in PERSON_METRIC_FILES}

print("Instantiating GNNDataset …")
gnn_ds = GNNDataset(
    pkl_path=DATA_PKL,
    split="train",
    seq_len=5,
    num_joints=28,
    coords_per_joint=3,
)

for sample in gnn_ds.data_list:
    for fn, mdict in metric_dicts.items():
        sample[fn] = mdict.get(sample["id"], 0.0)

# simple access check
example = gnn_ds.data_list[0]
assert all(fn in example for fn in PERSON_METRIC_FILES)
ok("GNNDataset integration: metrics attached without error")

# ---------------------------------------------------------------- SUMMARY
print("\nAll tests PASSED")


# ---------------------------------------------------------------- EXTRA COVERAGE REPORT
print("\n--- per-split coverage check ----------------------------------")
from sandbox.edge import convert_pkl_to_matrices

def load_person_metric(name):
    return pickle.load(open(os.path.join(METRIC_DIR, name), "rb"))

attn_dict = load_person_metric("attn_events_per_sample.pkl")
coll_dict = load_person_metric("collision_count_per_sample.pkl")

for split in ("train", "valid", "test"):
    try:
        ds_samples = convert_pkl_to_matrices(
            pkl_path='sandbox/action_dataset_joints_leg_sampled_5.pkl',
            spatial_pairs=[], seq_len=5, num_joints=28, coords_per_joint=3,
            split=split
        )
    except Exception:
        continue   # split may not exist
    ids = {s["id"] for s in ds_samples}
    attn_overlap = sum(1 for k in attn_dict if k in ids)
    coll_overlap = sum(1 for k in coll_dict if k in ids)
    print(f"{split:<5}: attention keys = {attn_overlap:4}/{len(attn_dict):4}  "
          f"collision keys = {coll_overlap:4}/{len(coll_dict):4}")
