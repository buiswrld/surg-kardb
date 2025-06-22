import os
import pickle
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from edge import (
    convert_pkl_to_matrices,
    get_spatial_pairs_from_named_joints,
)


_METRIC_FILES = {
    "attn": "attn_events_per_sample.pkl",
    "coll": "collision_count_per_sample.pkl",
    "group": "group_level_metrics.pkl",
}


class GNNDataset(Dataset):
    """PyTorch `Dataset` wrapper that returns a PyG `Data` object **plus**
    concatenated empirical‑metric features.
    """

    def __init__(
        self,
        pkl_path: str,
        *,
        split: str = "train",
        seq_len: int = 5,
        num_joints: int = 28,
        coords_per_joint: int = 3,
        metrics_dir: str = "./metrics",
    ) -> None:
        
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split: {split}")

        spatial_pairs = get_spatial_pairs_from_named_joints()
        self.data_list = convert_pkl_to_matrices(
            pkl_path=pkl_path,
            spatial_pairs=spatial_pairs,
            seq_len=seq_len,
            num_joints=num_joints,
            coords_per_joint=coords_per_joint,
            split=split,
        )

        # ---------------- metrics ---------------
        self.attn_dict = self._load_metric(metrics_dir, _METRIC_FILES["attn"])
        self.coll_dict = self._load_metric(metrics_dir, _METRIC_FILES["coll"])
        self.group_dict = self._load_metric(metrics_dir, _METRIC_FILES["group"])

        self._group_vec = torch.tensor(
            [
                float(self.group_dict.get("__GROUP_ATTENTION_FRAMES__", 0)),
                float(self.group_dict.get("__MEAN_CENTROID_DISPERSION__", 0)),
                float(self.group_dict.get("__TOTAL_CENTROID_DRIFT__", 0)),
                float(self.group_dict.get("__TOTAL_PAIRWISE_MOTION__", 0)),
            ],
            dtype=torch.float,
        )

    def _load_metric(self, metrics_dir: str, fname: str):
        path = os.path.join(metrics_dir, fname)
        if not os.path.exists(path):
            return {}
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        sample = self.data_list[idx]

        # ---------------- core graph data ----------------
        x = torch.tensor(sample["x"], dtype=torch.float)
        edge_index = sample["edge_index"].long()
        y = torch.tensor([sample["y"]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = sample.get("id", f"sample_{idx}")

        # ---------------- metrics tensor -----------------
        attn = float(self.attn_dict.get(data.id, 0))
        coll = float(self.coll_dict.get(data.id, 0))
        per_person_vec = torch.tensor([attn, coll], dtype=torch.float)

        data.metrics = torch.cat([per_person_vec, self._group_vec])
        # Keep raw scalars for convenience / analysis
        data.attn_events = attn
        data.collision_count = coll
        return data
