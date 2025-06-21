import os
import torch
import numpy as np
import pickle
import io
from tqdm import tqdm

def matches_blackbox_persp(file_path):
    try:
        code = file_path.split("/")[-1].split("_")[1]
        return code.endswith("004")
    except:
        return "" 

def find_matching_mp4_files(root_dir="/pasteur/results/ghent_surg/Post 2022"):
    final_list = []

    # Get all first-level directories in root_dir
    first_level_dirs = [
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    for directory in tqdm(first_level_dirs, desc="Processing directories"):
        # Get all .mp4 files in the current directory and subdirectories
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.mp4'):
                    file_path = os.path.join(root, file)

                    # Check if the file meets the criteria
                    if len(file_path) > 0 and matches_blackbox_persp(file_path):
                        final_list.append(file_path)
                        break 

    return final_list

class CPU_Unpickler(pickle.Unpickler):
  def find_class(self, module, name): 
    if module == 'torch.storage' and name == '_load_from_bytes':
      return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
    else:
      return super().find_class(module, name )

def read_pickle(path):
    with open(path, 'rb') as f:
      return CPU_Unpickler(f).load()


"""
Params: start_frame: initial pickle
        end_frame: final pickle
        dataset: path to dataset
        step: # frames to skip between 2 consecutive frames
Returns: array containing relevant frames, that have been converted from pickle to dict

If step size is too big, the final step will be concatinated to ensure the
  final frame is included
"""
def get_time_slice(start_frame, end_frame, dataset, step=1, full=True, debug=False):
  pickle_list = sorted(os.listdir(dataset))
  if debug:
    time_slice = pickle_list[:1000]
  elif full:
    time_slice = pickle_list
  else:
    time_slice = (pickle_list[pickle_list.index(start_frame):pickle_list.index(end_frame)+1 :step])
  for i in tqdm(range(len(time_slice)), desc="Reading files"): 
    time_slice[i] = read_pickle(os.path.join(dataset, time_slice[i]))
  return time_slice


"""
Applies pred_cam transformations to dataset
Params: data: frame
        flip_yz: if True displays top down view
Returns transformed joints and vertices
"""
def apply_transform(data, flip_yz=False):
    vertices = (data['vertices'] + data['pred_cam_t'].unsqueeze(1)).detach().cpu().numpy()
    joints = (data['joints3d'] + data['pred_cam_t'].unsqueeze(1)).detach().cpu().numpy()
    if flip_yz:
        vertices = vertices[..., [0, 2, 1]]
        joints = joints[..., [0, 2, 1]]
    return joints, vertices


"""
Params: frame_at_joints_3d: frame['joints3d']
        p_max: max amount of ppl in a frame
        j: num_joints in joints_3d
Returns: padded tensor of shape (p_max, j, 3)
"""
def apply_padding(frame_at_joints3d, p_max = 20, j = 127):
    p_actual = frame_at_joints3d.shape[0]

    # Move tensor to CPU if it's on CUDA
    if isinstance(frame_at_joints3d, torch.Tensor):
        frame_at_joints3d = frame_at_joints3d.detach().cpu().numpy()

    padded_tensor = np.zeros((p_max, j, 3))  # Initialize numpy array
    padded_tensor[:p_actual, ...] = frame_at_joints3d  # Copy data

    return padded_tensor.astype(np.float32)  # Ensure proper dtype


"""
Applies padding and transformation to every frame in timeslice
"""
def preprocess(time_slice):
  for frame in time_slice:
    frame['joints3d'], frame['vertices'] = apply_transform(frame)
    # frame['joints3d'] = apply_padding(frame['joints3d']) 
  return time_slice
