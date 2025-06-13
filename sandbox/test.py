from gcn_dataset import GNNDataset
import torch

# Use correct sequence length based on your dataset
pkl_path = "action_dataset_joints_leg_sampled_150.pkl"
exclude_group = ["left_ankle"]
seq_len = 150  # FIXED: was 5 before, but your data uses 150 time steps
num_joints = 28
coords_per_joint = 3

dataset = GNNDataset(
    pkl_path=pkl_path,
    split="train",
    seq_len=seq_len,
    num_joints=num_joints,
    coords_per_joint=coords_per_joint,
    exclude_groups=exclude_group
)

# Print one sample
sample = dataset[0]
x = sample.x
edge_index = sample.edge_index
label = sample.y
sample_id = sample.id

# Derive number of joints after exclusion
joints_after_exclusion = x.shape[0] // seq_len

print("\nSample ID:", sample_id)
print("Label:", label.item())
print("x shape:", x.shape)
print("Remaining joints per frame (after exclusion):", joints_after_exclusion)
print("Edge index shape:", edge_index.shape)
