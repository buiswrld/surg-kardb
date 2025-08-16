import numpy as np
import pickle
import os
import torch
import io
from empirical.util import read_pickle, apply_transform

# STAR 
def compute_gaze_vector(joints):
    """
    Computes the gaze direction vector for a single person's joint data.

    Returns: (origin, gaze_vector): A tuple where `origin` is the midpoint between the eyes,
                                    and `gaze_vector` is the normalized gaze direction.
    """
    L = joints[57]
    R = joints[56]
    N = joints[12]
    mu = (L + R) / 2  # Midpoint between eyes
    initial_gaze = mu - N
    # Define a plane using the vector between the eyes and the initial gaze
    eye_vector = L - R
    plane_normal = np.cross(eye_vector, initial_gaze)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    # Projection of the initial gaze onto the plane
    corrected_gaze = initial_gaze - np.dot(initial_gaze, plane_normal) * plane_normal
    corrected_gaze = corrected_gaze / np.linalg.norm(corrected_gaze)
    return mu, corrected_gaze

def count_focused_attention_events(data_dict, margin_of_error, time_frame, start_frame_dict, total_num_frames, step_size=3):
    """
    """
    events = {}
    window_size = time_frame 
    event_record = [0 for _ in range(total_num_frames)]
    
    for person_id, person_data in data_dict.items():
        start_frame = start_frame_dict[person_id]
        person_data = np.array(person_data)
        T = person_data.shape[0]

        # Compute gaze vectors (direction component) for each frame.
        gaze_directions = np.zeros((T, 3))
        for t in range(T):
            _, gaze_vector = compute_gaze_vector(person_data[t])
            gaze_directions[t] = gaze_vector
        event_count = 0
        
        # sliding window
        for t in range(0, T - window_size + 1, step_size):
            window = gaze_directions[t:t+window_size]
            ref_vector = window[0]
            # Compute the dot product between the first vector and each vector in the window. (clip just keeps bounds consistent for doing inverse cosine)
            dots = np.clip(np.sum(window * ref_vector, axis=1), -1.0, 1.0)
            # Calculate angle differences (in radians)
            angle_deviations_rad = np.arccos(dots)
            angle_deviations_deg = np.degrees(angle_deviations_rad) 
            # If all angles in this window are within the allowed margin, count an event
            if np.all(angle_deviations_deg <= margin_of_error):
                # print("event: time " + str(t))
                event_count += 1
                for frame_idx in range(start_frame, start_frame + window_size): #Update event record 
                    event_record[frame_idx] += 1
        events[person_id] = event_count
        
    return events, event_record 

def process_files(folder_path, debug=False):
    #returns dictionary where keys are IDs and values are in the shape [T, J, 3]
    paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')])
    if debug:
        paths = paths[4930:]
    total_num_frames = len(paths) 
    data = {}
    start_frame_dict = {} 
    for frame_idx, path in enumerate(paths):
        f = read_pickle(path)
        joints, vertices = apply_transform(f, True)
        for i, tracker_id in enumerate(f['trackers']):
            person_joints = joints[i]
            if tracker_id in data:
                data[tracker_id].append(person_joints)
            else:
                data[tracker_id] = [person_joints]
                start_frame_dict[tracker_id] = frame_idx   
    return data, start_frame_dict, total_num_frames

if __name__ == "__main__":
    print("starting")