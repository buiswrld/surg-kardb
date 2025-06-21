import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import math
import random
import os
import io
import time
from empirical.util import CPU_Unpickler, read_pickle, apply_transform, get_time_slice, preprocess, apply_padding

def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def detect_engagement(wrist_threshold, elbow_threshold, joints_3d):
    """
    Detect engagement based on wrist and elbow distances.

    Args:
        wrist_threshold (float): Maximum allowed distance between wrists.
        elbow_threshold (float): Maximum allowed distance between elbows.
        joints_3d (list): List of joint positions for a person.

    Returns:
        bool: True if the person is engaged, False otherwise.
    """
    # Extract wrist and elbow joint positions
    left_wrist, right_wrist = joints_3d[20], joints_3d[21]
    left_elbow, right_elbow = joints_3d[18], joints_3d[19]

    # Compute distances
    wrist_distance = euclidean_distance(left_wrist, right_wrist)
    elbow_distance = euclidean_distance(left_elbow, right_elbow)

    # Check if both wrist and elbow distances are below their respective thresholds
    return wrist_distance < wrist_threshold and elbow_distance < elbow_threshold
 

def detect_engagement_event(time_slice,
                            wrist_threshold, 
                            elbow_threshold, 
                            event_time_threshold
    ):
    """
    Detect engagement events across frames.

    Args:
        start_frame (int): Beginning frame.
        end_frame (int): Ending frame.
        dataset (str): Path to dataset directory.
        wrist_threshold (float): Wrist distance threshold.
        elbow_threshold (float): Elbow distance threshold.
        event_time_threshold (int): Minimum consecutive engagement frames.
        step (int): Number of frames to skip per iteration.

    Returns:
        dict: Engagement events per person.
        int: Total events in the frame
    """
    # Initialize storage structures
    engagement_events = {}  # Stores frames where engagement occurs per person
    id_dict = {}  # Tracks persons' last known joint positions

    # Iterate through frames
    for frame_idx, frame in enumerate(time_slice):
        joints_3d_list = frame['joints3d']
        tracker_ids = frame['trackers']

        # Iterate through persons in the frame
        for j in range(len(joints_3d_list)):
            if j >= len(tracker_ids):
                continue  # Skip this iteration if out of bounds
            person_id = tracker_ids[j]
            joints_3d = joints_3d_list[j]
            
            # Add person to dictionary if first occurrence
            if person_id not in id_dict:
                id_dict[person_id] = joints_3d[0]

            # Detect engagement
            if detect_engagement(wrist_threshold, elbow_threshold, joints_3d):
                if person_id not in engagement_events:
                    engagement_events[person_id] = []
                engagement_events[person_id].append(frame_idx)  # Store frame index
                # print(f"Engagement detected for Person {person_id} at Frame {frame_idx}")

    # Find engagement sequences
    event_counts = {}
    total_events = 0
    event_record = [0 for _ in range(len(time_slice))]
    for person_id, frames in engagement_events.items():
        frames.sort()  # Ensure frames are in order
        consecutive_count = 1

        for i in range(1, len(frames)):
            if frames[i] == frames[i - 1] + 1 and consecutive_count < event_time_threshold:
                consecutive_count += 1
            else:
                if consecutive_count >= event_time_threshold:
                    event_counts[person_id] = event_counts.get(person_id, 0) + 1
                    total_events += 1
                    for j in range(frames[i]-consecutive_count+1, frames[i]+1): #TODO ~ Fix Logic 
                        event_record[j] = event_record[j] + 1
                consecutive_count = 1  # Reset count

        # Check for the last sequence
        if consecutive_count >= event_time_threshold:
            event_counts[person_id] = event_counts.get(person_id, 0) + 1
            total_events += 1
            for j in range(frames[i]-consecutive_count+1, frames[i]+1):
                event_record[j] = event_record[j] + 1

    return event_counts, total_events, event_record


def tester(): 
    # Parameters for detection
    start_frame = 'frame_000000.pkl'
    end_frame = 'frame_049640.pkl' # ex: frame_049640 is our final frame of the video, for 4960 pkl frames total
    wrist_threshold = 0.3
    elbow_threshold = 0.6 # TODO: Can try 0.7 or 0.8 
    event_time_threshold = 15 # 5 seconds, adjust this value based on how many frames represent a second
    step = 1  # Process every frame
    
    time_start = time.time()
    # Run the function to detect engagement events
    engagement_events, total_events, event_record = detect_engagement_event(
        start_frame=start_frame,
        end_frame=end_frame,
        dataset="/Users/jamesignacio/Downloads/HMR Test Folder 2/joint_out/",
        wrist_threshold=wrist_threshold,
        elbow_threshold=elbow_threshold,
        event_time_threshold=event_time_threshold,
        step=step
    )
    time_end = time.time()

    print(f"Execution time: {time_end - time_start} seconds")
    # Output the result
    print("Engagement events detected:", engagement_events)
    print("Total events:", total_events)
    print("Event record:", len(event_record))

def pipeline_tester():
    dataset = "/pasteur/data/ghent_surg/full_hmr_outputs/220610_22011/joint_out/"
    time_slice = get_time_slice(0, 0, dataset, step=1, debug=True)
    time_slice = preprocess(time_slice)  

    wrist_threshold = 0.3
    elbow_threshold = 0.6 # TODO: Can try 0.7 or 0.8 
    event_time_threshold = 15 # 5 seconds, adjust this value based on how many frames represent a second
    
    time_start = time.time()
    # Run the function to detect engagement events
    engagement_events, total_events, event_record = detect_engagement_event(
        time_slice=time_slice,
        wrist_threshold=wrist_threshold,
        elbow_threshold=elbow_threshold,
        event_time_threshold=event_time_threshold
    )
    print(event_record)
    time_end = time.time()
    print(f"Execution time: {time_end - time_start} seconds")

if __name__ == "__main__":
    pipeline_tester()