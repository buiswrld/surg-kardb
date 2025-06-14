import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from edge import convert_pkl_to_matrices, get_spatial_pairs_from_named_joints
import pickle


class GNNDataset(Dataset):
    def __init__(self, pkl_path, split="train", seq_len=5, num_joints=28, coords_per_joint=3, exclude_groups=[]):
        """
        Args:
            pkl_path (str): Path to .pkl file containing {'train': [...], 'valid': [...], 'test': [...]}
            split (str): One of 'train', 'valid', or 'test'
            seq_len (int): Number of time steps
            num_joints (int): Number of joints per frame
            coords_per_joint (int): Usually 3 for (x, y, z)
        """
        assert split in ["train", "valid", "test"], f"Invalid split: {split}"

        self.exclude_groups = exclude_groups  # [ADDED]
        self.num_joints = num_joints


        spatial_pairs = get_spatial_pairs_from_named_joints()

        self.data_list = convert_pkl_to_matrices(
            pkl_path=pkl_path,
            spatial_pairs=spatial_pairs,
            seq_len=seq_len,
            num_joints=num_joints,
            coords_per_joint=coords_per_joint,
            split=split
        )

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]

        x = torch.tensor(sample["x"], dtype=torch.float)
        x = self.reshape_joints(x)  # shape (T, 28, 3)

        if not self.exclude_groups.empty():
            from joints import MAIN_JOINTS
            excluded_joints = set(self.exclude_groups)  # e.g., {"left_ankle"}
            joint_indices = [i for i, j in enumerate(MAIN_JOINTS) if j not in excluded_joints]
            x = x[:, joint_indices, :]

        x = x.reshape(-1, 3)

        edge_index = sample["edge_index"].long()
        y = torch.tensor([sample["y"]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = sample.get("id", f"sample_{idx}")
        return data

    
    def reshape_joints(self, input_array):
        length = input_array.shape[0] // self.num_joints
        return input_array.reshape(length, self.num_joints, -1)
