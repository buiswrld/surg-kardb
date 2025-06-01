import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from edge import convert_pkl_to_matrices, get_spatial_pairs_from_named_joints
import pickle


class GNNDataset(Dataset):
    def __init__(self, pkl_path, split="train", seq_len=5, num_joints=28, coords_per_joint=3):
        """
        Args:
            pkl_path (str): Path to .pkl file containing {'train': [...], 'valid': [...], 'test': [...]}
            split (str): One of 'train', 'valid', or 'test'
            seq_len (int): Number of time steps
            num_joints (int): Number of joints per frame
            coords_per_joint (int): Usually 3 for (x, y, z)
        """
        assert split in ["train", "valid", "test"], f"Invalid split: {split}"
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
        edge_index = torch.tensor(sample["edge_index"], dtype=torch.long)
        y = torch.tensor([sample["y"]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = sample.get("id", f"sample_{idx}")
        return data
