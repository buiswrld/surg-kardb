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
        edge_index = sample["edge_index"].long()
        y = torch.tensor([sample["y"]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = sample.get("id", f"sample_{idx}")
        return data
    
    def reshape_joints(self, input_array):  # [ADDED]
        length = input_array.shape[0] if input_array.shape != (84,) else 1
        return input_array.reshape(length, 28, 3)