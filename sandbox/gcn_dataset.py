import os, pickle, torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from edge import convert_pkl_to_matrices, get_spatial_pairs_from_named_joints

_METRIC_KEYS = {
    "dist"  : ["total_distance"],
    "speed" : ["speed_mean", "speed_std"],
    "engage": ["engagement_events"],
    "attn"  : ["attention_changes"],
    "motion": ["total_distance", "speed_mean", "speed_std"],
    "usage" : ["engagement_events", "attention_changes"],
    "all"   : [
        "total_distance", "speed_mean", "speed_std",
        "engagement_events", "gaze_stability",
        "attention_changes", "window_seconds"
    ],
}


class GNNDataset(Dataset):
    """
    Converts the pre-baked .pkl dataset into PyG Data objects *and*
    appends a configurable empirical-metrics vector under `.metrics`.
    """

    def __init__(
        self,
        pkl_path: str,
        *,
        split: str = "train",
        seq_len: int = 5,
        num_joints: int = 28,
        coords_per_joint: int = 3,
        metrics_path: str = "./metrics/per_clip_metrics_1s.pkl",
        metric_set: str = "all",
    ) -> None:

        if split not in {"train", "valid", "test"}:
            raise ValueError(f"Invalid split '{split}'")
        if metric_set not in _METRIC_KEYS:
            raise ValueError(f"Unknown metric_set '{metric_set}'")

        spatial_pairs = get_spatial_pairs_from_named_joints()
        self.data_list = convert_pkl_to_matrices(
            pkl_path=pkl_path,
            spatial_pairs=spatial_pairs,
            seq_len=seq_len,
            num_joints=num_joints,
            coords_per_joint=coords_per_joint,
            split=split,
        )

        with open(metrics_path, "rb") as f:
            full_metric_dict = pickle.load(f)
        self.metric_keys = _METRIC_KEYS[metric_set]
        self.metric_vec  = torch.tensor(
            [float(full_metric_dict.get(k, 0.0)) for k in self.metric_keys],
            dtype=torch.float,
        )
        self.metrics_dim = len(self.metric_keys)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        sample = self.data_list[idx]

        x = torch.tensor(sample["x"], dtype=torch.float)
        edge_index = sample["edge_index"].long()
        y = torch.tensor([sample["y"]], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.id = sample.get("id", f"sample_{idx}")
        data.metrics         = self.metric_vec.clone().unsqueeze(0)
        data.total_distance  = self.metric_vec[self.metric_keys.index("total_distance")] \
                               if "total_distance" in self.metric_keys else None
        data.engagement_cnt  = self.metric_vec[self.metric_keys.index("engagement_events")] \
                               if "engagement_events" in self.metric_keys else None
        data.attn_switches   = self.metric_vec[self.metric_keys.index("attention_changes")] \
                               if "attention_changes" in self.metric_keys else None
        return data
