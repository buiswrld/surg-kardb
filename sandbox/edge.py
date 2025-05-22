import pickle
import torch
import numpy as np
from typing import List, Tuple

def get_spatial_pairs_from_named_joints() -> List[Tuple[int, int]]:
    joint_names = [
        'pelvis', 'left_hip', 'right_hip',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
        'head', 'jaw', 'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear',
        'left_shoulder', 'right_shoulder', 'left_collar', 'right_collar', 'neck',
        'spine1', 'spine2', 'spine3',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_foot', 'right_foot'
    ]
    name_to_idx = {name: i for i, name in enumerate(joint_names)}

    return [
        (name_to_idx['pelvis'], name_to_idx['left_hip']),
        (name_to_idx['pelvis'], name_to_idx['right_hip']),
        (name_to_idx['pelvis'], name_to_idx['spine1']),
        (name_to_idx['spine1'], name_to_idx['spine2']),
        (name_to_idx['spine2'], name_to_idx['spine3']),
        (name_to_idx['spine3'], name_to_idx['neck']),
        (name_to_idx['left_shoulder'], name_to_idx['left_elbow']),
        (name_to_idx['left_elbow'], name_to_idx['left_wrist']),
        (name_to_idx['right_shoulder'], name_to_idx['right_elbow']),
        (name_to_idx['right_elbow'], name_to_idx['right_wrist']),
        (name_to_idx['left_hip'], name_to_idx['left_knee']),
        (name_to_idx['left_knee'], name_to_idx['left_ankle']),
        (name_to_idx['left_ankle'], name_to_idx['left_foot']),
        (name_to_idx['right_hip'], name_to_idx['right_knee']),
        (name_to_idx['right_knee'], name_to_idx['right_ankle']),
        (name_to_idx['right_ankle'], name_to_idx['right_foot']),
        (name_to_idx['neck'], name_to_idx['left_collar']),
        (name_to_idx['neck'], name_to_idx['right_collar']),
        (name_to_idx['left_collar'], name_to_idx['left_shoulder']),
        (name_to_idx['right_collar'], name_to_idx['right_shoulder']),
        (name_to_idx['neck'], name_to_idx['jaw']),
        (name_to_idx['jaw'], name_to_idx['head']),
        (name_to_idx['head'], name_to_idx['nose']),
        (name_to_idx['head'], name_to_idx['left_eye']),
        (name_to_idx['head'], name_to_idx['right_eye']),
        (name_to_idx['head'], name_to_idx['left_ear']),
        (name_to_idx['head'], name_to_idx['right_ear']),
    ]


def convert_pkl_to_matrices(
    pkl_path: str,
    spatial_pairs: List[Tuple[int, int]],
    seq_len: int,
    num_joints: int = 28,
    coords_per_joint: int = 3
):
    """
    Load a .pkl dataset of joint sequences and convert each sample into:
    - A node feature matrix (flattened joint coordinates)
    - A combined edge index from spatial and temporal connections

    Args:
        pkl_path (str): Path to the .pkl file.
        spatial_pairs (List[Tuple[int, int]]): List of spatial joint connections.
        seq_len (int): Sequence length per sample.
        num_joints (int): Number of joints per frame.
        coords_per_joint (int): Usually 3 (x, y, z).

    Returns:
        List[dict]: Each dict contains 'x', 'edge_index', 'y', 'id'
    """
    def get_spatial_edges(num_joints: int, seq_len: int, pairs: List[Tuple[int, int]]) -> torch.Tensor:
        edges = []
        for t in range(seq_len):
            for i, j in pairs:
                src = t * num_joints + i
                dst = t * num_joints + j
                edges.append((src, dst))
                edges.append((dst, src))
        return torch.tensor(edges, dtype=torch.long).t()

    def get_temporal_edges(num_joints: int, seq_len: int) -> torch.Tensor:
        edges = []
        for t in range(seq_len - 1):
            for j in range(num_joints):
                src = t * num_joints + j
                dst = (t + 1) * num_joints + j
                edges.append((src, dst))
                edges.append((dst, src))
        return torch.tensor(edges, dtype=torch.long).t()

    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)

    results = []
    for sample_array, label, identifier in dataset["train"]:
        assert sample_array.shape == (seq_len, num_joints * coords_per_joint), \
            f"Expected ({seq_len}, {num_joints * coords_per_joint}), got {sample_array.shape}"

        reshaped = sample_array.reshape(seq_len, num_joints, coords_per_joint)
        node_features = reshaped.reshape(seq_len * num_joints, coords_per_joint)

        spatial_edges = get_spatial_edges(num_joints, seq_len, spatial_pairs)
        temporal_edges = get_temporal_edges(num_joints, seq_len)
        edge_index = torch.cat([spatial_edges, temporal_edges], dim=1)

        results.append({
            "x": node_features,         # shape (T*J, 3)
            "edge_index": edge_index,   # shape (2, N_edges)
            "y": int(label),            # class label
            "id": identifier            # sample identifier
        })

    return results



def main():

    pkl_path = "action_dataset_joints_leg_sampled_150.pkl"
    seq_len = 150
    num_joints = 28
    coords_per_joint = 3
    spatial_pairs = get_spatial_pairs_from_named_joints()

    results = convert_pkl_to_matrices(
        pkl_path=pkl_path,
        spatial_pairs=spatial_pairs,
        seq_len=seq_len,
        num_joints=num_joints,
        coords_per_joint=coords_per_joint
    )

    sample = results[0]
    x = sample["x"]
    edge_index = sample["edge_index"]
    y = sample["y"]
    identifier = sample["id"]

    print("Sample loaded and reshaped")
    print(f"ID: {identifier}")
    print(f"Node features shape: {x.shape} (expected {(seq_len*num_joints, coords_per_joint)})")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Label: {y}")

    # first 5 joints at frame 0
    print("\nFirst 5 joints at frame 0:")
    for j in range(5):
        joint_vec = x[j]
        print(f"  Joint {j}: [x={joint_vec[0]:.4f}, y={joint_vec[1]:.4f}, z={joint_vec[2]:.4f}]")

    #10 edges
    print("\nFirst 10 edges:")
    for i in range(min(10, edge_index.shape[1])):
        src, dst = edge_index[:, i]
        print(f"  Edge {i}: {src.item()} → {dst.item()}")

    # automated checks
    assert x.shape == (seq_len * num_joints, coords_per_joint)
    assert edge_index.shape[0] == 2
    assert edge_index.max() < seq_len * num_joints, "Edge index has out-of-bound node index"

if __name__ == "__main__":
    main()