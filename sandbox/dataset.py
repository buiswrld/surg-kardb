import pandas as pd 
import os
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

import copy
from skspatial.objects import Point, Vector
import pickle
import matplotlib.colors as mcolors
from tqdm import tqdm 
from joints import (pelvic_indices, arm_indices, head_indices, thorax_indices, leg_indices, spine_indices,
                    pelvic_indices_pose, arm_indices_pose, head_indices_pose, thorax_indices_pose, leg_indices_pose, spine_indices_pose,
                    head_struct_indices, eye_indices, ear_indices, elbow_indices, wrist_indices, SMPL_JOINT_NAMES, JOINT_NAMES) 

import torch
from PIL import Image
import torchvision.transforms as T
import random
from downstream import * 

class MixerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split):
        self.dataset = read_pickle(dataset_path)[split]
    
    def __len__(self):
        return len(self.dataset) 
    
    def __getitem__(self, index):
        dataset_example = self.dataset[index] 
        output = {} 
        output['label'] = dataset_example[1] 
        output['embedding_seq'] = dataset_example[0] 
        return output  

#Gnerate training splits 
def generate_splits(total_length):
    train_size = int(0.7 * total_length)
    valid_size = int(0.15 * total_length)
    test_size = total_length - train_size - valid_size

    # Create a list of labels representing the split
    labels = ["train"] * train_size + ["valid"] * valid_size + ["test"] * test_size

    # Shuffle the labels to randomize the split
    random.shuffle(labels)
    return labels 

def shuffle_dict(input): 
    keys = list(input.keys())
    random.shuffle(keys)
    return {key: input[key] for key in keys}

#Preprocessing of the dataset 
def dataset_preprocess(source_path, labels_to_exclude=[3.0], binary=False): #eft with 0 and 2
    #Define ablation dataset pkl subsets (joints, pose, shape, positions)
    dataset = read_pickle(source_path) 
    dataset = {key: value for key, value in dataset.items() if not (value[1] in labels_to_exclude)} #Remove large insruemnt label 

    #Separate C and E video subsets 
    train_split_cx = {} 
    valtest_split_ex = {} 
    for k, v in dataset.items():
        video_key = k.split("_")[0][0].lower() 
        if video_key == "c":
            train_split_cx[k] = v 
        else:
            valtest_split_ex[k] = v 
    
    #Shuffle both dictionaries 
    train_split_cx = shuffle_dict(train_split_cx)
    valtest_split_ex = shuffle_dict(valtest_split_ex)
    

    #Put into tuple format with splits 
    output_dataset = {
        "train" : [],
        "valid" : [],
        "test" : [] 
    }

    #All c1 examples are assigned to train 
    for k, v in train_split_cx.items():
        output_dataset['train'].append((v[0], v[1], k))

    #Each E example has an equal probability of being assigned to val or test
    for k, v in valtest_split_ex.items(): 
        if random.random() > 0.5: 
            output_dataset['valid'].append((v[0], v[1], k))
        else:
            output_dataset['test'].append((v[0], v[1], k))
        
    write_pickle(output_dataset, source_path) 

#Define constants
metadata_root_dir = "/pasteur/u/bencliu/baseline/experiments/simulation/metrics"
mixer_results_save_dir = " /pasteur/u/bencliu/baseline/experiments/simulation/mixer_results"
DS_EXP_PATH = "/pasteur/u/bencliu/baseline/data/datasets/experiments/downstream_subset/"
CORE_EXP_PATH = "/pasteur/u/bencliu/baseline/experiments/simulation/"
metadata_root_dir_labels = "/pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata"

#HELPER FUNCTIONS 
def write_pickle(data_object, path):
    with open(path, 'wb') as handle:
        pickle.dump(data_object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding='latin1')

def general_wrapper():
    metadata_save_dir = "/pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/ablation"
    prepare_dataset(frame_sample=1, metadata_save_dir=metadata_save_dir, joint_ablation=True, desired_len=50)
    breakpoint() 
    prepare_dataset(frame_sample=1, metadata_save_dir=metadata_save_dir, joint_ablation=True, desired_len=25)
    prepare_dataset(frame_sample=1, metadata_save_dir=metadata_save_dir, joint_ablation=True, desired_len=10)
    prepare_dataset(frame_sample=1, metadata_save_dir=metadata_save_dir, joint_ablation=True, desired_len=5)
    breakpoint() 
    prepare_dataset(frame_sample=1, metadata_save_dir=metadata_save_dir, joint_ablation=True, desired_len=150)
    prepare_dataset(frame_sample=1, metadata_save_dir=metadata_save_dir, joint_ablation=True, desired_len=125)
    prepare_dataset(frame_sample=1, metadata_save_dir=metadata_save_dir, joint_ablation=True, desired_len=100)
    prepare_dataset(frame_sample=1, metadata_save_dir=metadata_save_dir, joint_ablation=True, desired_len=75)
    breakpoint() 
    prepare_dataset_binary(frame_sample=3, metadata_save_dir=metadata_save_dir)
    breakpoint() 

#MAIN Function - Generating Dataset 
def prepare_dataset_binary(frame_sample=1, metadata_save_dir="/pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata"):
    #Dictionary of example id => properties, labels 
    action_labels_path = os.path.join(metadata_root_dir_labels, "ghent_simulation_action_labels_v2.csv")
    labels = pd.read_csv(action_labels_path) 
    labels = labels.dropna(subset=['min_start'])
    labels = labels[labels['type'] == 'full'].reset_index(drop=True)

    #Define files for specific joints   
    path_vision_arm = os.path.join(metadata_save_dir, "binary_dataset_vision_arm_sampled_" + str(frame_sample) + ".pkl")
    dict_vision_arm = {} 
    path_vision = os.path.join(metadata_save_dir, "binary_dataset_vision_sampled_" + str(frame_sample) + ".pkl")
    dict_vision = {} 
    path_arm = os.path.join(metadata_save_dir, "binary_dataset_arm_sampled_" + str(frame_sample) + ".pkl")
    dict_arm = {} 
    path_wrists = os.path.join(metadata_save_dir, "binary_dataset_wrists_sampled_" + str(frame_sample) + ".pkl")
    dict_wrists = {} 
    path_elbows = os.path.join(metadata_save_dir, "binary_dataset_wrists_elbows_sampled_" + str(frame_sample) + ".pkl")
    dict_elbows = {} 
    path_eye = os.path.join(metadata_save_dir, "binary_dataset_wrists_elbows_eyes_sampled_" + str(frame_sample) + ".pkl")
    dict_eye = {} 
    path_head = os.path.join(metadata_save_dir, "binary_dataset_wrists_elbows_eyes_head_sampled_" + str(frame_sample) + ".pkl")
    dict_head = {} 
    path_ear = os.path.join(metadata_save_dir, "binary_dataset_wrists_elbows_eyes_head_ear_sampled_" + str(frame_sample) + ".pkl")
    dict_ear = {} 
    
    total_seq_count = 0 

    exp_names = ["e1", "e2", "e4", "c1", "c2", "c4"]
    for exp_name in exp_names: #Loop through videos 
        #Read in metadata for each vieo 
        exp_metadata_dir = os.path.join(CORE_EXP_PATH, "metrics", exp_name)
        master_metric_dict_path = os.path.join(exp_metadata_dir, "master_metadata.pkl") 
        master_metric_dict = read_pickle(master_metric_dict_path)
        labels_subset = labels[labels['video'] == exp_name].reset_index(drop=True)

        #Loop through action labels 
        for i, row in tqdm(labels_subset.iterrows(), total=len(labels_subset)):
            tracklet = row['tracklet']
            total_frame_start = row['frame_start']  #TODO TROUBLESHOOT
            total_seq = row['total_seq'] 
            total_seq_count += total_seq
            action_label = row['label']
            if tracklet not in master_metric_dict:
                print("exception") 
                continue 
            maximum_frames = len(master_metric_dict[tracklet]["joints_3d"])

            for j in range(total_seq): #Loop through full frame sequence for each row 
                #Retieve frame indices for 5-seocnd clip 
                frame_start = total_frame_start + (j * 150) #150 frames for 5 second interval
                frame_end = frame_start + 150  #5 frame buffer sequence 
                example_key = exp_name + "_" + str(tracklet) + "_" + str(frame_start) + "_" + str(action_label) 

                #Gather input metadata 
                joints_3d = np.array(master_metric_dict[tracklet]["joints_3d"][frame_start:frame_end])
                if not frame_end < maximum_frames:
                    if total_seq - j - 1 > 0:
                        print(exp_name, tracklet) 
                    break 

                #Process frame sampling
                if frame_sample > 1:
                    joints_3d = joints_3d[::frame_sample]

                wrists_joints = joints_3d[:, wrist_indices, :]
                elbow_joints = joints_3d[:, elbow_indices, :]
                eye_joints = joints_3d[:, eye_indices, :]
                head_joints = joints_3d[:, head_struct_indices, :]
                ear_joints = joints_3d[:, ear_indices, :]

                wrists_joints = wrists_joints.reshape(wrists_joints.shape[0], -1)
                elbow_joints = elbow_joints.reshape(elbow_joints.shape[0], -1)
                eye_joints = eye_joints.reshape(eye_joints.shape[0], -1)
                head_joints = head_joints.reshape(head_joints.shape[0], -1)
                ear_joints = ear_joints.reshape(ear_joints.shape[0], -1)
                
                #Create concatenations for full ablations -- 
                arm_input = np.concatenate([wrists_joints, elbow_joints], axis=1) 
                vision_input = np.concatenate([eye_joints, head_joints, ear_joints], axis=1) 
                vision_arm_input = np.concatenate([vision_input, arm_input], axis=1) 
                wrist_input = wrists_joints
                elbow_input = np.concatenate([wrists_joints, elbow_joints], axis=1) 
                eye_input = np.concatenate([wrists_joints, elbow_joints, eye_joints], axis=1) 
                head_input = np.concatenate([wrists_joints, elbow_joints, eye_joints, head_joints], axis=1) 
                ear_input = np.concatenate([wrists_joints, elbow_joints, eye_joints, head_joints, ear_joints], axis=1) 

                dict_vision_arm[example_key] = [vision_arm_input, action_label]
                dict_vision[example_key] = [vision_input, action_label] 
                dict_arm[example_key] = [arm_input, action_label] 
                dict_wrists[example_key] = [wrist_input, action_label] 
                dict_elbows[example_key] = [elbow_input, action_label] 
                dict_eye[example_key] = [eye_input, action_label] 
                dict_head[example_key] = [head_input, action_label]
                dict_ear[example_key] = [ear_input, action_label]

    write_pickle(dict_vision_arm, path_vision_arm)
    write_pickle(dict_vision, path_vision)
    write_pickle(dict_arm, path_arm)
    write_pickle(dict_wrists, path_wrists)
    write_pickle(dict_elbows, path_elbows)
    write_pickle(dict_eye, path_eye)
    write_pickle(dict_head, path_head)
    write_pickle(dict_ear, path_ear)

    dataset_paths = [ path_vision_arm, path_vision, path_arm, path_wrists,
                     path_elbows, path_eye, path_head, path_ear]
    for ds_path in dataset_paths:
        dataset_preprocess(ds_path, labels_to_exclude=[3.0, 1.0], binary=True) 
    print([x.split("/")[-1] for x in dataset_paths])
    breakpoint() 

#MAIN Function - Generating Dataset 
def prepare_dataset(frame_sample=1, metadata_save_dir="/pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/seq_ablation", 
                    joint_ablation=False, render=False, desired_len=150):
    #Dictionary of example id => properties, labels 
    action_labels_path = os.path.join(metadata_root_dir_labels, "ghent_simulation_action_labels_v2.csv")
    labels = pd.read_csv(action_labels_path) 
    labels = labels.dropna(subset=['min_start'])
    labels = labels[labels['type'] == 'full'].reset_index(drop=True)

    #Define files for joint individuals 
    path_vision_arm = os.path.join(metadata_save_dir, "action_dataset_vision_arm_sampled_" + str(desired_len) + ".pkl")
    dict_vision_arm = {} 
    path_vision = os.path.join(metadata_save_dir, "action_dataset_vision_sampled_" + str(desired_len) + ".pkl")
    dict_vision = {} 
    path_arm = os.path.join(metadata_save_dir, "action_dataset_arm_sampled_" + str(desired_len) + ".pkl")
    dict_arm = {} 
    path_wrists = os.path.join(metadata_save_dir, "action_dataset_wrists_sampled_" + str(desired_len) + ".pkl")
    dict_wrists = {} 
    path_elbows = os.path.join(metadata_save_dir, "action_dataset_wrists_elbows_sampled_" + str(desired_len) + ".pkl")
    dict_elbows = {} 
    path_eye = os.path.join(metadata_save_dir, "action_dataset_wrists_elbows_eyes_sampled_" + str(desired_len) + ".pkl")
    dict_eye = {} 
    path_head = os.path.join(metadata_save_dir, "action_dataset_wrists_elbows_eyes_head_sampled_" + str(desired_len) + ".pkl")
    dict_head = {} 
    path_ear = os.path.join(metadata_save_dir, "action_dataset_wrists_elbows_eyes_head_ear_sampled_" + str(desired_len) + ".pkl")
    dict_ear = {} 

    #Define files for joint classes 
    action_dataset_joints_pelvis_path = os.path.join(metadata_save_dir, "action_dataset_joints_pelvis_sampled_" + str(desired_len) + ".pkl")
    action_dataset_joints_pelvis_dict = {} 
    action_dataset_joints_arm_path = os.path.join(metadata_save_dir, "action_dataset_joints_arm_sampled_" + str(desired_len) + ".pkl")
    action_dataset_joints_arm_dict = {} 
    action_dataset_joints_head_path = os.path.join(metadata_save_dir, "action_dataset_joints_head_sampled_" + str(desired_len) + ".pkl")
    action_dataset_joints_head_dict = {} 
    action_dataset_joints_thorax_path = os.path.join(metadata_save_dir, "action_dataset_joints_thorax_sampled_" + str(desired_len) + ".pkl")
    action_dataset_joints_thorax_dict = {}
    action_dataset_joints_spine_path = os.path.join(metadata_save_dir, "action_dataset_joints_spine_sampled_" + str(desired_len) + ".pkl")
    action_dataset_joints_spine_dict = {}  
    action_dataset_joints_leg_path = os.path.join(metadata_save_dir, "action_dataset_joints_leg_sampled_" + str(desired_len) + ".pkl")
    action_dataset_joints_leg_dict = {} 
    total_seq_count = 0 

    #Define new files for pose 
    action_dataset_pose_pelvis_path = os.path.join(metadata_save_dir, "action_dataset_pose_pelvis_sampled_" + str(desired_len) + ".pkl")
    action_dataset_pose_pelvis_dict = {} 
    action_dataset_pose_arm_path = os.path.join(metadata_save_dir, "action_dataset_pose_arm_sampled_" + str(desired_len) + ".pkl")
    action_dataset_pose_arm_dict = {} 
    action_dataset_pose_head_path = os.path.join(metadata_save_dir, "action_dataset_pose_head_sampled_" + str(desired_len) + ".pkl")
    action_dataset_pose_head_dict = {} 
    action_dataset_pose_thorax_path = os.path.join(metadata_save_dir, "action_dataset_pose_thorax_sampled_" + str(desired_len) + ".pkl")
    action_dataset_pose_thorax_dict = {}
    action_dataset_pose_spine_path = os.path.join(metadata_save_dir, "action_dataset_pose_spine_sampled_" + str(desired_len) + ".pkl")
    action_dataset_pose_spine_dict = {}  
    action_dataset_pose_leg_path = os.path.join(metadata_save_dir, "action_dataset_pose_leg_sampled_" + str(desired_len) + ".pkl")
    action_dataset_pose_leg_dict = {} 

    #Define files for joint individuals V2 with pelvis 
    path_wrists_v2 = os.path.join(metadata_save_dir, "action_dataset_pelvis_wrists_sampled_" + str(desired_len) + ".pkl")
    dict_wrists_v2 = {} 
    path_elbows_v2 = os.path.join(metadata_save_dir, "action_dataset_pelvis_wrists_elbows_sampled_" + str(desired_len) + ".pkl")
    dict_elbows_v2 = {} 
    path_eye_v2 = os.path.join(metadata_save_dir, "action_dataset_pelvis_wrists_elbows_eyes_sampled_" + str(desired_len) + ".pkl")
    dict_eye_v2 = {} 
    path_head_v2 = os.path.join(metadata_save_dir, "action_dataset_pelvis_wrists_elbows_eyes_head_sampled_" + str(desired_len) + ".pkl")
    dict_head_v2 = {} 
    path_ear_v2 = os.path.join(metadata_save_dir, "action_dataset_pelvis_wrists_elbows_eyes_head_ear_sampled_" + str(desired_len) + ".pkl")
    dict_ear_v2 = {} 

    attn_switch_dict = {
        "walk" : [],
        "tool" : [],
        "observe" : [],
        "instrument" : [],
    }

    hand_dist_dict = {
        "walk" : [],
        "tool" : [],
        "observe" : [],
        "instrument" : [],
    }

    action_label_keys = {
        "0.0" : "tool",
        "1.0" : "walk",
        "2.0" : "observe", 
        "3.0" : "instrument",
    }


    exp_names = ["e1", "e2", "e4", "c1", "c2", "c4"]
    for exp_name in exp_names: #Loop through videos 
        #Read in metadata for each video 
        exp_metadata_dir = os.path.join(CORE_EXP_PATH, "metrics", exp_name)
        master_metric_dict_path = os.path.join(exp_metadata_dir, "master_metadata.pkl") 
        master_metric_dict = read_pickle(master_metric_dict_path)
        exp_metadata_clip_dir = os.path.join(CORE_EXP_PATH, "tracklet_metrics", exp_name)

        #Match action labels to vidoe 
        labels_subset = labels[labels['video'] == exp_name].reset_index(drop=True)

        #Establish bounds for plotting 
        hip_joint_index = 0
        tracker_metadata = dict(sorted(master_metric_dict.items()))
        x_min, x_max, y_min, y_max = 0, 0, 0, 0 
        pelvis_points = [] 
        for tracker_key, tracker_metadata_dict in tqdm(tracker_metadata.items()): 
            joints_3d = tracker_metadata_dict['joints_3d']
            xyz_positions = [joints[hip_joint_index] for joints in joints_3d]
            pelvis_points.extend(xyz_positions) 
        x_min = min([pos[0] for pos in pelvis_points])
        x_max = max([pos[0] for pos in pelvis_points])
        y_min = min([pos[2] for pos in pelvis_points])
        y_max = max([pos[2] for pos in pelvis_points])
        counter = 0 

        #Loop through action labels 
        for i, row in tqdm(labels_subset.iterrows(), total=len(labels_subset)):
            tracklet = row['tracklet']
            total_frame_start = row['frame_start'] * 30 #Bug in the original dataset file -- frame_start is the second started 
            total_seq = row['total_seq'] 
            total_seq_count += total_seq
            action_label = row['label']
            if tracklet not in master_metric_dict:
                print("exception") 
                continue 
            maximum_frames = len(master_metric_dict[tracklet]["joints_3d"])

            for j in range(total_seq): #Loop through full frame sequence for each row 
                #Retieve frame indices for 5-seocnd clip 
                counter += 1 
                frame_start = total_frame_start + (j * 150) #150 frames for 5 second interval
                frame_end = frame_start + 150  #5 frame buffer sequence 
                example_key = exp_name + "_" + str(tracklet) + "_" + str(frame_start) + "_" + str(action_label) 

                #Gather input metadata 
                joints_3d = np.array(master_metric_dict[tracklet]["joints_3d"][frame_start:frame_end])
                pose = np.array(master_metric_dict[tracklet]["pose"][frame_start:frame_end])
                shape = np.array(master_metric_dict[tracklet]["shape"][frame_start:frame_end])
                frame_ids = np.array(master_metric_dict[tracklet]["frame_ids"][frame_start:frame_end])
                if not frame_end < maximum_frames:
                    if total_seq - j - 1 > 0:
                        print(exp_name, tracklet) 
                    break 
                
                if desired_len < 150:
                    reference_len = 150 
                    step_size = reference_len / desired_len
                    sampled_indices = [int(i * step_size) for i in range(desired_len)]
                    sampled_indices[-1] = reference_len - 1  # Ensure the last index is included
                    joints_3d = joints_3d[sampled_indices]
                    pose = pose[sampled_indices]
                    shape = shape[sampled_indices]
                    frame_ids = frame_ids[sampled_indices]

                #Process frame sampling
                if frame_sample > 1:
                    joints_3d = joints_3d[::frame_sample]
                    pose = pose[::frame_sample]
                    shape = shape[::frame_sample]
                    frame_ids = frame_ids[::frame_sample]

                #JOINT CLASS ABLATION 
                # -------------------
                pelvic_joints = joints_3d[:, pelvic_indices, :]
                arm_joints = joints_3d[:, arm_indices, :]
                head_joints = joints_3d[:, head_indices, :]
                thorax_joints = joints_3d[:, thorax_indices, :]
                spine_joints = joints_3d[:, spine_indices, :]
                leg_joints = joints_3d[:, leg_indices, :]

                pelvic_joints = pelvic_joints.reshape(pelvic_joints.shape[0], -1)
                arm_joints = arm_joints.reshape(arm_joints.shape[0], -1)
                head_joints = head_joints.reshape(head_joints.shape[0], -1)
                thorax_joints = thorax_joints.reshape(thorax_joints.shape[0], -1)
                spine_joints = spine_joints.reshape(spine_joints.shape[0], -1)
                leg_joints = leg_joints.reshape(leg_joints.shape[0], -1)
                
                #Create concatenations for full ablations 
                pelvis_input = pelvic_joints
                arm_input = np.concatenate([pelvic_joints, arm_joints], axis=1) 
                head_input = np.concatenate([pelvic_joints, arm_joints, head_joints], axis=1) 
                thorax_input = np.concatenate([pelvic_joints, arm_joints, head_joints, thorax_joints], axis=1) 
                spine_input = np.concatenate([pelvic_joints, arm_joints, head_joints, thorax_joints, spine_joints], axis=1) 
                leg_input = np.concatenate([pelvic_joints, arm_joints, head_joints, thorax_joints, spine_joints, leg_joints], axis=1) 

                action_dataset_joints_pelvis_dict[example_key] = [pelvis_input, action_label]
                action_dataset_joints_arm_dict[example_key] = [arm_input, action_label] 
                action_dataset_joints_head_dict[example_key] = [head_input, action_label] 
                action_dataset_joints_thorax_dict[example_key] = [thorax_input, action_label] 
                action_dataset_joints_spine_dict[example_key] = [spine_input, action_label] 
                action_dataset_joints_leg_dict[example_key] = [leg_input, action_label] 

                # print('JOINTS')
                #print("SEQ LEN: ", pelvis_input.shape[0])
                # print("Pelvis: ", pelvis_input.shape[1])
                # print("Arm: ", arm_input.shape[1])
                # print("Head: ", head_input.shape[1])
                # print("Thorax: ", thorax_input.shape[1])
                # print("Spine: ", spine_input.shape[1])
                # print("Leg: ", leg_input.shape[1])
                
                #JOINT INDIVIDUAL ABLATION 
                # ------------------------
                pelvic_joints = joints_3d[:, pelvic_indices, :]
                wrists_joints = joints_3d[:, wrist_indices, :]
                elbow_joints = joints_3d[:, elbow_indices, :]
                eye_joints = joints_3d[:, eye_indices, :]
                head_joints = joints_3d[:, head_struct_indices, :]
                ear_joints = joints_3d[:, ear_indices, :]

                pelvic_joints = pelvic_joints.reshape(pelvic_joints.shape[0], -1)
                wrists_joints = wrists_joints.reshape(wrists_joints.shape[0], -1)
                elbow_joints = elbow_joints.reshape(elbow_joints.shape[0], -1)
                eye_joints = eye_joints.reshape(eye_joints.shape[0], -1)
                head_joints = head_joints.reshape(head_joints.shape[0], -1)
                ear_joints = ear_joints.reshape(ear_joints.shape[0], -1)
                
                #Create concatenations for full ablations  
                arm_input = np.concatenate([wrists_joints, elbow_joints], axis=1) 
                vision_input = np.concatenate([eye_joints, head_joints, ear_joints], axis=1) 
                vision_arm_input = np.concatenate([vision_input, arm_input], axis=1) 
                wrist_input = wrists_joints
                elbow_input = np.concatenate([wrists_joints, elbow_joints], axis=1) 
                eye_input = np.concatenate([wrists_joints, elbow_joints, eye_joints], axis=1) 
                head_input = np.concatenate([wrists_joints, elbow_joints, eye_joints, head_joints], axis=1) 
                ear_input = np.concatenate([wrists_joints, elbow_joints, eye_joints, head_joints, ear_joints], axis=1) 

                dict_vision_arm[example_key] = [vision_arm_input, action_label]
                dict_vision[example_key] = [vision_input, action_label] 
                dict_arm[example_key] = [arm_input, action_label] 
                dict_wrists[example_key] = [wrist_input, action_label] 
                dict_elbows[example_key] = [elbow_input, action_label] 
                dict_eye[example_key] = [eye_input, action_label] 
                dict_head[example_key] = [head_input, action_label]
                dict_ear[example_key] = [ear_input, action_label]

                # print("INDIV JOINTS")
                # print("Arm: ", arm_input.shape[1])
                # print("Vision: ", vision_input.shape[1])
                # print("Vision_Arm: ", vision_arm_input.shape[1])
                # print("Wrist: ", wrist_input.shape[1])
                # print("Elbow: ", elbow_input.shape[1])
                # print("Eye: ", eye_input.shape[1])
                # print("Head: ", head_input.shape[1])
                # print("Ear: ", ear_input.shape[1])

                 #JOINT INDIVIDUAL ABLATION V2 w/ PELVIS
                # ------------------------

                wrist_input_v2 = np.concatenate([pelvic_joints, wrists_joints], axis=1) 
                elbow_input_v2 = np.concatenate([pelvic_joints, wrists_joints, elbow_joints], axis=1) 
                eye_input_v2 = np.concatenate([pelvic_joints, wrists_joints, elbow_joints, eye_joints], axis=1) 
                head_input_v2 = np.concatenate([pelvic_joints, wrists_joints, elbow_joints, eye_joints, head_joints], axis=1) 
                ear_input_v2 = np.concatenate([pelvic_joints, wrists_joints, elbow_joints, eye_joints, head_joints, ear_joints], axis=1) 

                dict_wrists_v2[example_key] = [wrist_input_v2, action_label] 
                dict_elbows_v2[example_key] = [elbow_input_v2, action_label] 
                dict_eye_v2[example_key] = [eye_input_v2, action_label] 
                dict_head_v2[example_key] = [head_input_v2, action_label]
                dict_ear_v2[example_key] = [ear_input_v2, action_label]

                # print("INDIV JOINTS")
                # print("Wrist: ", wrist_input_v2.shape[1])
                # print("Elbow: ", elbow_input_v2.shape[1])
                # print("Eye: ", eye_input_v2.shape[1])
                # print("Head: ", head_input_v2.shape[1])
                # print("Ear: ", ear_input_v2.shape[1])

                #POSE ABLATION REPRESENTATION 
                # --------------------------
                pelvic_poses = pose[:, pelvic_indices_pose, :] 
                arm_poses = pose[:, arm_indices_pose, :]
                head_poses = pose[:, head_indices_pose, :]
                thorax_poses = pose[:, thorax_indices_pose, :]
                spine_poses = pose[:, spine_indices_pose, :]
                leg_poses = pose[:, leg_indices_pose, :]

                pelvic_poses = pelvic_poses.reshape(pelvic_poses.shape[0], -1)
                arm_poses = arm_poses.reshape(arm_poses.shape[0], -1)
                head_poses = head_poses.reshape(head_poses.shape[0], -1)
                thorax_poses = thorax_poses.reshape(thorax_poses.shape[0], -1)
                spine_poses = spine_poses.reshape(spine_poses.shape[0], -1)
                leg_poses = leg_poses.reshape(leg_poses.shape[0], -1)
                
                #Create concatenations for full ablations 
                pelvis_input = pelvic_poses
                arm_input = np.concatenate([pelvic_poses, arm_poses], axis=1) 
                head_input = np.concatenate([pelvic_poses, arm_poses, head_poses], axis=1) 
                thorax_input = np.concatenate([pelvic_poses, arm_poses, head_poses, thorax_poses], axis=1) 
                spine_input = np.concatenate([pelvic_poses, arm_poses, head_poses, thorax_poses, spine_poses], axis=1) 
                leg_input = np.concatenate([pelvic_poses, arm_poses, head_poses, thorax_poses, spine_poses, leg_poses], axis=1) 

                action_dataset_pose_pelvis_dict[example_key] = [pelvis_input, action_label]
                action_dataset_pose_arm_dict[example_key] = [arm_input, action_label] 
                action_dataset_pose_head_dict[example_key] = [head_input, action_label] 
                action_dataset_pose_thorax_dict[example_key] = [thorax_input, action_label] 
                action_dataset_pose_spine_dict[example_key] = [spine_input, action_label] 
                action_dataset_pose_leg_dict[example_key] = [leg_input, action_label] 

                # print('POSE')
                # print("Pelvis: ", pelvis_input.shape[1])
                # print("Arm: ", arm_input.shape[1])
                # print("Head: ", head_input.shape[1])
                # print("Thorax: ", thorax_input.shape[1])
                # print("Spine: ", spine_input.shape[1])
                # print("Leg: ", leg_input.shape[1])
                
                #Render graphs for all clips 
                if render:
                    action_str = action_label_keys[str(action_label)]
                    clip_metric_dir = os.path.join(exp_metadata_clip_dir, "t" + str(tracklet), str(frame_start) + "_" + str(frame_end) + "_" + action_str) 
                    if not os.path.exists(clip_metric_dir):
                        os.makedirs(clip_metric_dir, exist_ok=True)

                    #Extract tracklet metadata stratfiied by frames 
                    xyz_positions = [joints[hip_joint_index] for joints in joints_3d.tolist()]

                    #Create flow and heat maps 
                    helper_tracklet_flow_heat_maps(xyz_positions, clip_metric_dir, 
                                                    x_bounds=(x_min, x_max), y_bounds=(y_min, y_max))

                    #Compute distance traversal map 
                    helper_dist_traversal_graph(xyz_positions, clip_metric_dir)

                    #Compute 3D attention maps and 1D attention state map 
                    attn_switch_state_vector = helper_tracklet_attention_maps(joints_3d, clip_metric_dir)

                    #Compute stats on average wrist distances and number of attention switches 
                    lw_index = JOINT_NAMES.index("left_wrist")
                    rw_index = JOINT_NAMES.index("right_wrist")
                    left_wrists = joints_3d[:, lw_index, :].reshape(joints_3d.shape[0], -1).tolist()
                    right_wrists = joints_3d[:, rw_index, :].reshape(joints_3d.shape[0], -1).tolist() 
                    wrist_dist_vec = []
                    for lw, rw in zip(left_wrists, right_wrists):
                        wrist_dist = euclidean_distance_3d(lw, rw) 
                        wrist_dist_vec.append(wrist_dist) 
                    avg_wrist_dist = sum(wrist_dist_vec) / len(wrist_dist_vec)
                    attn_switch = sum(attn_switch_state_vector)
                    attn_switch_dict[action_str].append(attn_switch) 
                    hand_dist_dict[action_str].append(avg_wrist_dist) 
        print(exp_name, counter) 
    
    if render:
        write_pickle(attn_switch_dict, "/pasteur/u/bencliu/baseline/experiments/simulation/tracklet_metrics/attn_switch_dict.pkl")
        write_pickle(hand_dist_dict, "/pasteur/u/bencliu/baseline/experiments/simulation/tracklet_metrics/hand_dist_dict.pkl")

    #Save joint class paths 
    write_pickle(action_dataset_joints_pelvis_dict, action_dataset_joints_pelvis_path)
    write_pickle(action_dataset_joints_arm_dict, action_dataset_joints_arm_path)
    write_pickle(action_dataset_joints_head_dict, action_dataset_joints_head_path)
    write_pickle(action_dataset_joints_thorax_dict, action_dataset_joints_thorax_path)
    write_pickle(action_dataset_joints_spine_dict, action_dataset_joints_spine_path)
    write_pickle(action_dataset_joints_leg_dict, action_dataset_joints_leg_path)

    #Save joint individuals 
    write_pickle(dict_vision_arm, path_vision_arm)
    write_pickle(dict_vision, path_vision)
    write_pickle(dict_arm, path_arm)
    write_pickle(dict_wrists, path_wrists)
    write_pickle(dict_elbows, path_elbows)
    write_pickle(dict_eye, path_eye)
    write_pickle(dict_head, path_head)
    write_pickle(dict_ear, path_ear)

    #Save joint individuals V2 w/ PELVIS
    write_pickle(dict_wrists_v2, path_wrists_v2)
    write_pickle(dict_elbows_v2, path_elbows_v2)
    write_pickle(dict_eye_v2, path_eye_v2)
    write_pickle(dict_head_v2, path_head_v2)
    write_pickle(dict_ear_v2, path_ear_v2)

    #Save pose ablations 
    write_pickle(action_dataset_pose_pelvis_dict, action_dataset_pose_pelvis_path)
    write_pickle(action_dataset_pose_arm_dict, action_dataset_pose_arm_path)
    write_pickle(action_dataset_pose_head_dict, action_dataset_pose_head_path)
    write_pickle(action_dataset_pose_thorax_dict, action_dataset_pose_thorax_path)
    write_pickle(action_dataset_pose_spine_dict, action_dataset_pose_spine_path)
    write_pickle(action_dataset_pose_leg_dict, action_dataset_pose_leg_path)


    dataset_paths = [action_dataset_joints_thorax_path, action_dataset_joints_spine_path, action_dataset_joints_head_path,
                     action_dataset_joints_arm_path, action_dataset_joints_pelvis_path, action_dataset_joints_leg_path,
                     path_vision_arm, path_vision, path_arm, path_wrists, path_elbows, path_eye, path_head, path_ear,
                     action_dataset_pose_pelvis_path, action_dataset_pose_arm_path, action_dataset_pose_head_path, 
                     action_dataset_pose_thorax_path, action_dataset_pose_spine_path, action_dataset_pose_leg_path,
                     path_wrists_v2, path_elbows_v2, path_eye_v2, path_head_v2, path_ear_v2]
    for ds_path in dataset_paths:
        dataset_preprocess(ds_path) 
    print([x.split("/")[-1] for x in dataset_paths])


#Referenced from GPT 
def generate_sinusoidal_encoding(positions, max_frame_count, d_model=64):
    """
    Generate sinusoidal positional encodings.

    Args:
    - positions: numpy array of shape (num_positions, 1)
    - max_frame_count: maximum frame count
    - d_model: dimensionality of the positional encoding

    Returns:
    - positional_encodings: numpy array of shape (num_positions, d_model)
    """
    num_positions = positions.shape[0]
    angles = np.arange(d_model) / d_model
    angles = 1 / (np.power(10000, (2 * (angles // 2)) / np.float32(d_model)))
    
    # Apply sine to even indices in the array
    angles[:, 0::2] = np.sin(positions * angles[:, 0::2])
    # Apply cosine to odd indices in the array
    angles[:, 1::2] = np.cos(positions * angles[:, 1::2])

    return angles

def debugging():
    geometry_analysis_wrapper("c2")
    geometry_analysis_wrapper("e1") 
    breakpoint()  

#Adapted from downstream code file 
def geometry_analysis_high_wrapper():
    letters = ['e', 'c'] 
    nums = range(1, 5) 
    for letter in letters:
        for num in nums:
            exp_name = letter + str(num) 
            geometry_analysis_wrapper(exp_name) 

#Process HMR Geometry Metadata Outputs 
def geometry_analysis_wrapper(name=""):
    #Initialize save paths and core directories
    tracker_metadata = {} 
    joint_metadata_path = os.path.join(CORE_EXP_PATH, "joint_out", name)
    exp_metadata_dir = os.path.join(CORE_EXP_PATH, "metrics", name)
    if not os.path.exists(exp_metadata_dir):
        os.makedirs(exp_metadata_dir, exist_ok=True)
    master_metric_dict_path = os.path.join(exp_metadata_dir, "master_metadata.pkl") 

    #Attain frame paths 
    files = os.listdir(joint_metadata_path)
    pkl_files = [os.path.join(joint_metadata_path, file) for file in files if file.endswith('.pkl')]
    frame_paths = sorted(pkl_files)

    #Process into tracklet-level [~10 minute runtime] 
    for frame_path in tqdm(frame_paths):
        frame_metadata = read_pickle(frame_path) 
        current_frame = int(frame_path.split("/")[-1].split(".")[0])  
        joints_3d = frame_metadata['joints3d'].cpu().tolist() #.tolist() (num_people, 127, 3) 
        trackers = frame_metadata['trackers'] 
        pred_pose = frame_metadata['pred_pose'].cpu().tolist() # (num_people, 22, 3, 3)
        #pred_vert = frame_metadata['vertices'].cpu().tolist()
        pred_shape = frame_metadata['pred_shape'].cpu().tolist()

        for i, tracker_id in enumerate(frame_metadata['trackers']):
            if tracker_id not in tracker_metadata:
                tracker_metadata[tracker_id] = {
                    "joints_3d" : [joints_3d[i]],
                    "trackers" : [trackers[i]],
                    "pose" : [pred_pose[i]],  #"vert" : [pred_vert[i]], 
                    "shape" : [pred_shape[i]],
                    "frame_ids" : [current_frame]
                }
            else: 
                tracker_metadata[tracker_id]['joints_3d'].append(joints_3d[i]) 
                tracker_metadata[tracker_id]['trackers'].append(trackers[i]) 
                tracker_metadata[tracker_id]['pose'].append(pred_pose[i]) 
                #tracker_metadata[tracker_id]['vert'].append(pred_vert[i]) 
                tracker_metadata[tracker_id]['shape'].append(pred_shape[i]) 
                tracker_metadata[tracker_id]['frame_ids'].append(current_frame) 
    
    write_pickle(tracker_metadata, master_metric_dict_path)
    
""" 
Helper functions
""" 

if __name__ == "__main__":
    print("Starting dataaset loader") 
    general_wrapper() 
    breakpoint() 
    dataset_preprocess()
    breakpoint() 
    test_path = "/pasteur/u/bencliu/baseline/experiments/simulation/mixer_metadata/test.pkl"
    ds = MixerDataset(test_path, "valid")   
    breakpoint() 
    

""" 
Embedding Dimensions

#Print embedding dimensions
print("Joints Only: ", joints_input.shape[1])
print("Pose All: ", joints_input.shape[1])
print("Shape All: ", joints_input.shape[1])
print("Full All: ", joints_input.shape[1])

print("General Joints: ", general_joints.shape[1])
print("Lower Joints: ", lower_joints.shape[1])
print("Head Joints: ", head_joints.shape[1])
print("Arm Joints: ", arm_joints.shape[1])
print("Abdominal Joints: ", abdominal_joints.shape[1])

___

Print embedding dimensions
print("Joints All: ", joints_input.shape[1])
print("Pose All: ", pose_input.shape[1])
print("Shape All: ", shape_input.shape[1])
print("Full All: ", full_input.shape[1])

print("Pelvis: ", pelvis_input.shape[1])
print("Arm: ", arm_input.shape[1])
print("Head: ", head_input.shape[1])
print("Thorax: ", thorax_input.shape[1])
print("Spine: ", spine_input.shape[1])
print("Leg: ", leg_input.shape[1])

print("Arm: ", arm_input.shape[1])
print("Vision: ", vision_input.shape[1])
print("Vision_Arm: ", vision_arm_input.shape[1])
print("Wrist: ", wrist_input.shape[1])
print("Elbow: ", elbow_input.shape[1])
print("Eye: ", eye_input.shape[1])
print("Head: ", head_input.shape[1])
print("Ear: ", ear_input.shape[1])

"""