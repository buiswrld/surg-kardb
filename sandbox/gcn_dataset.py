import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from edge import convert_pkl_to_matrices, get_spatial_pairs_from_named_joints


class GNNDataset(Dataset):
    def __init__(self, pkl_path, seq_len=5, num_joints=28, coords_per_joint=3):
        spatial_pairs = get_spatial_pairs_from_named_joints()
        self.data_list = convert_pkl_to_matrices(
            pkl_path=pkl_path,
            spatial_pairs=spatial_pairs,
            seq_len=seq_len,
            num_joints=num_joints,
            coords_per_joint=coords_per_joint
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


def main():
    dataset = GNNDataset("action_dataset_joints_leg_sampled_5.pkl")
    
    for i in range(3):
        data = dataset[i]
        print(f"Sample {i}:")
        print(f"  x shape        : {data.x.shape}")         # (T × J, 3)
        print(f"  edge_index     : {data.edge_index.shape}")  # (2, num_edges)
        print(f"  y              : {data.y.item()}")
        print(f"  id             : {data.id}")

if __name__ == "__main__":
    main()
