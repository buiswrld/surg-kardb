import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
import random
import io
import os
import time
from empirical.util import CPU_Unpickler, read_pickle, apply_transform, get_time_slice, preprocess, apply_padding

def get_pelvis(joints):
    """
    Extract the pelvis joint from the joints array.
    """
    return joints[:, 0, (0, 2)]  # Extract the pelvis joint for all people in the frame

# USEFUL
def calculate_velocity(prev_position, curr_position):
    """
    Calculate the velocity between two positions.
    """
    return np.linalg.norm(curr_position - prev_position)

# USEFUL
def is_walking(velocity, velocity_threshold=0.1):
    """
    Determine if a person is walking based on velocity.
    """
    return velocity > velocity_threshold

def detect_collisions(frames, radius, velocity_threshold):
    """
    Detect collision events based on bubble overlap and walking status.
    
    Args:
        frames (list): List of frames, where each frame is a dictionary containing 'joints3d' and 'trackers'.
        radius (float): Radius of each person's bubble.
        velocity_threshold (float): Threshold to determine if a person is walking.
    
    Returns:
        collision_events (list): A list of collision events, where each event is a dictionary containing:
            - frame_idx: Index of the frame where the collision occurred.
            - person_1: ID of the first person involved in the collision.
            - person_2: ID of the second person involved in the collision.
            - distance: Distance between the two people's pelvis joints.
            - walking_status: Tuple indicating if person_1 and person_2 were walking.
    """
    frame_collisions = [] 
    collision_events = []
    total_collisions = 0
    prev_pelvis = None  # Store pelvis positions from the previous frame
    prev_trackers = None  # Store trackers from the previous frame

    for frame_idx, frame in enumerate(frames):
        frame_collisions_i = 0 
        joints3d = frame['joints3d']
        trackers = frame['trackers']
        pelvis = get_pelvis(joints3d)  # Get pelvis joints for all people in the current frame

        # Initialize walking status for the current frame
        walking_status = {tracker: False for tracker in trackers}

        # Calculate walking status for each person
        if frame_idx > 0:
            for i, tracker in enumerate(trackers):
                # If the tracker existed in the previous frame, calculate velocity and check walking status
                if prev_trackers and tracker in prev_trackers:
                    velocity = calculate_velocity(prev_pelvis[prev_trackers.index(tracker)], pelvis[i])
                    walking_status[tracker] = is_walking(velocity, velocity_threshold)

        # Check for bubble overlaps (collisions)
        for i in range(len(trackers)):
            for j in range(i + 1, len(trackers)):
                distance = np.linalg.norm(pelvis[i] - pelvis[j])
                if distance < 2 * radius:  # Sum of radii
                    # Check if either person was walking
                    frame_collisions_i += 1 
                    if walking_status[trackers[i]] or walking_status[trackers[j]]:
                        total_collisions += 1
                        collision_events.append({
                            'frame_idx': frame_idx,
                            'person_1': trackers[i],
                            'person_2': trackers[j],
                            'distance': distance,
                            'walking_status': (walking_status[trackers[i]], walking_status[trackers[j]])
                        })
        # Update previous pelvis positions and trackers for the next frame
        prev_pelvis = pelvis
        prev_trackers = trackers
        frame_collisions.append(frame_collisions_i)

    return collision_events, total_collisions, frame_collisions

def execute_collision_detection(dataset, start_frame, end_frame, step, radius, v_thres):
    """
    Execute collision detection on a time slice of the dataset.
    
    Args:
        dataset (str): Path to the dataset directory.
        start_frame (int): Start frame index.
        end_frame (int): Stop frame index.
        time_slice (list): List of frames to process.
        radius (float): Radius of each person's bubble.
        v_thres (float): Threshold to determine if a person is walking.
    
    Returns:
        collision_events (list): A list of collision events, as described in detect_collisions().
        total_collisions (int): Total number of collisions detected.
    """
    time_slice = get_time_slice(start_frame, end_frame, dataset, step)
    time_slice = preprocess(time_slice)  
    collision_events, total_collisions = detect_collisions(time_slice, radius, v_thres)
    return collision_events, total_collisions

# Example usage:
def tester():
    time_start = time.time()

    collision_events, total_collisions = execute_collision_detection(dataset = "/Users/jamesignacio/Downloads/HMR Test Folder 2/joint_out/",
                                                                    start_frame='frame_000000.pkl',
                                                                    end_frame='frame_049640.pkl',
                                                                    step=1,
                                                                    radius=0.25, #Shoulder-to-shoulder 
                                                                    v_thres=2)

    time_end = time.time()
    print(f"Execution time: {time_end - time_start} seconds")
    print(collision_events)
    print(f"Total Collisions: {total_collisions}")

def pipeline_tester():
    dataset = get_time_slice(0, 0, "/pasteur/data/ghent_surg/full_hmr_outputs/220610_22011_test_5000/joint_out/", debug=True)
    dataset = preprocess(dataset)
    collision_events, total_collisions, frame_collisions = detect_collisions(dataset, 0.25, 2) 
    print(len(collision_events))
    print(total_collisions)
    print(frame_collisions)
    breakpoint()

if __name__ == "__main__":
    pipeline_tester()