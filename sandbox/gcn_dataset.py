from edge import convert_pkl_to_matrices, get_spatial_pairs_from_named_joints

def main():
    pkl_path = "action_dataset_joints_leg_sampled_5.pkl"
    seq_len = 5
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
    print(f"ID: {sample['id']}")
    print(f"Label: {sample['y']}")
    print(f"Node features shape: {sample['x'].shape}")
    print(f"Edge index shape: {sample['edge_index'].shape}")

if __name__ == "__main__":
    main()
