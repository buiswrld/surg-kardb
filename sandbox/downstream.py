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


DS_EXP_PATH = "/pasteur/u/bencliu/baseline/data/datasets/experiments/downstream_subset/"
OUTPUT_FIGURE_PATH = "/pasteur/u/bencliu/baseline/data/datasets/experiments/downstream"
CORE_EXP_PATH = "/pasteur/u/bencliu/baseline/experiments/simulation/"

def geometry_analysis_high_wrapper():
    letters = ['e', 'c'] 
    nums = range(1, 5) 
    for letter in letters:
        for num in nums:
            exp_name = letter + str(num) 
            #geometry_analysis_wrapper(exp_name) 
            print(exp_name)
            geometry_analysis_trackers(exp_name)
            breakpoint() 

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
        joints_3d = frame_metadata['joints3d'] #.tolist() (num_people, 127, 3) 
        trackers = frame_metadata['trackers'] 
        pred_pose = frame_metadata['pred_pose'] # (num_people, 22, 3, 3)
        breakpoint() #TODO ~ repurpose saving technique here for each frame

        for i, tracker_id in enumerate(frame_metadata['trackers']):
            if tracker_id not in tracker_metadata:
                tracker_metadata[tracker_id] = {
                    "joints_3d" : [joints_3d[i]],
                    "trackers" : [trackers[i]],
                    "pose" : [pred_pose[i]], 
                    "frame_ids" : []
                }
            else: 
                tracker_metadata[tracker_id]['joints_3d'].append(joints_3d[i]) 
                tracker_metadata[tracker_id]['trackers'].append(trackers[i]) 
                tracker_metadata[tracker_id]['pose'].append(pred_pose[i]) 
                tracker_metadata[tracker_id]['frame_ids'].append(current_frame) 
    
    write_pickle(tracker_metadata, master_metric_dict_path)

def geometry_analysis_trackers(name=""):
    exp_metadata_dir = os.path.join(CORE_EXP_PATH, "metrics", name)
    master_metric_dict_path = os.path.join(exp_metadata_dir, "master_metadata.pkl") 
    tracker_metadata = read_pickle(master_metric_dict_path)
    hip_joint_index = 0
    tracker_metadata = dict(sorted(tracker_metadata.items()))

    #Find xyz bounds based on pelvis joints exclusively 
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
    print("Extracted scene bounds") 

    #Process each individual tracklet sequence and derive metrics 
    for tracker_key, tracker_metadata_dict in tqdm(tracker_metadata.items()):
        tracker_metric_dir = os.path.join(exp_metadata_dir, "t" + str(tracker_key))
        if not os.path.exists(tracker_metric_dir):
            os.makedirs(tracker_metric_dir, exist_ok=True)

        #Extract tracklet metadata stratfiied by frames 
        joints_3d = tracker_metadata_dict['joints_3d'] # (127, 3) 
        trackers = tracker_metadata_dict['trackers']
        pred_pose = tracker_metadata_dict['pose']
        frames = tracker_metadata_dict['frame_ids']
        xyz_positions = [joints[hip_joint_index] for joints in joints_3d]
        pelvis_np = np.array(xyz_positions)[::5]
        breakpoint() 

        #Create flow and heat maps 
        helper_tracklet_flow_heat_maps(xyz_positions, tracker_metric_dir, 
                                       x_bounds=(x_min, x_max), y_bounds=(y_min, y_max))

        #Compute distance traversal map 
        helper_dist_traversal_graph(xyz_positions, tracker_metric_dir)

        #Compute 3D attention maps and 1D attention state map 
        helper_tracklet_attention_maps(joints_3d, tracker_metric_dir)
        
def helper_tracklet_flow_heat_maps(xyz_positions, tracker_metric_dir, x_bounds, y_bounds):
    save_path_heatmap = os.path.join(tracker_metric_dir, "heatmap.png")
    save_path_flowmap = os.path.join(tracker_metric_dir, "flowmap.png")
    xy_positions = [[pos[0], pos[2]] for pos in xyz_positions]
    plot_heatmap_and_save(xy_positions, save_path_heatmap, 
                            x_bounds=x_bounds, y_bounds=y_bounds)
    plot_trajectory_and_save(xy_positions, save_path_flowmap, 
                            x_bounds=x_bounds, y_bounds=y_bounds)
    
def helper_tracklet_attention_maps(joints_3d, tracker_metric_dir, max_distance=1.8, fov=120):
    save_path_1d_map = os.path.join(tracker_metric_dir, "1d_state_attn.png")
    save_path_3d_map = os.path.join(tracker_metric_dir, "3d_attn.png")
    vision_joints = [] 
    for joint_set in joints_3d:
        joint_set = joint_set #.tolist() 
        vision_joints.append([joint_set[56], joint_set[57], joint_set[58], 
                              joint_set[59], joint_set[55], joint_set[12]])  

    vision_angles = []
    simlarity_stats = [] 
    origins = []
    attn_switch_state_vector = [] 
    last_direction = np.array([]) 
    for i, vision in enumerate(vision_joints):
        reye, leye, rear, lear, nose, neck = vision 
        eye_mid = (np.array(reye) + np.array(leye))/2
        a, b, c, _ = get_plane(np.array(rear), np.array(lear), np.array(neck), anchor=eye_mid)
        view_direction = np.array([a, b, c]) / np.linalg.norm(np.array([a, b, c]))
        if not is_in_cone(eye_mid, view_direction, nose, max_distance, fov)[0]:
            view_direction = -view_direction
        
        #Similarity - Approximate 1/3 second for attn switch 
        if last_direction.shape[0] > 0 and (i % 25 == 0):
            cos_sim = cosine_similarity(view_direction, last_direction)
            simlarity_stats.append(cos_sim) 
            attn_switch_state_vector.append(cos_sim < 0.75) #Change in 40 degrees 
            last_direction = view_direction  
        
        #Add to running lists and update 
        origins.append(eye_mid.tolist())
        vision_angles.append(view_direction) 
        if i == 0:
            last_direction = view_direction  
    
    save_3d_vectors_plot(vision_angles, origins, save_path_3d_map)
    save_1d_state_plot(attn_switch_state_vector, save_path_1d_map)
    return attn_switch_state_vector


def helper_dist_traversal_graph(xyz_positions, tracker_metric_dir):
    save_path_flowmap = os.path.join(tracker_metric_dir, "dist_graph.png")
    xy_positions = [[pos[0], pos[2]] for pos in xyz_positions]

    distances = [] 
    last_position = xy_positions[0]
    for i, curr_position in enumerate(xy_positions[1:]):
        curr_distance =  euclidean_distance(last_position, curr_position)
        distances.append(curr_distance)
        last_position = curr_position
    
    cumu_sum = cumulative_sum(distances) 
    plot_superimposed_bar_line_save(distances, cumu_sum, save_path_flowmap, dist_limit=0.01)

#Archived Function
def analysis_wrapper(name="c4_v2"):
    joint_metadata_path = "/pasteur/u/bencliu/baseline/data/datasets/experiments/crowd/joint_out"
    joint_metadata_path = os.path.join(joint_metadata_path, name)
    tracklets_core_save_path = "/pasteur/u/bencliu/baseline/data/datasets/experiments/downstream_subset"
    tracklets_core_path = os.path.join(tracklets_core_save_path, name + "_tracklets_core.pkl")
    tracklets_core = read_pickle(tracklets_core_path)
    sorted_keys = sorted(tracklets_core, key=lambda k: len(tracklets_core[k]['positions']), reverse=True)

    #Loop through top 10 tracklets 
    for i in range(10): #TOAD
        print("STARTING NEW TRACKLET: ", i) 
        print("____________________")
        top_tracklet = tracklets_core[sorted_keys[i]]
        tracklet_index = sorted_keys[i]

        # #Plot distances [COMPLETE]
        tracklet_distance(top_tracklet['positions'], tracklet_index, i) 
        print("Completed positions") 

        # #Plot trajectories 
        tracklet_trajectories(top_tracklet['positions'], tracklet_index, i) 
        print("Completed trajectories")

        # #Plot arms [COMPLETE]
        tracklet_arms(top_tracklet['arms'], tracklet_index, i) 
        print("Completed arm stats")

        # #Plot fingers 
        tracklet_hands(top_tracklet['fingers'], tracklet_index, i) 
        print("Completed finger stats")

        #Plot visions 
        tracklet_vision(top_tracklet['vision'], tracklet_index, i) 
        print("Completed vision stats")


#HELPER FUNCTIONS 
def write_pickle(data_object, path):
    with open(path, 'wb') as handle:
        pickle.dump(data_object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle, encoding='latin1')


def plot_trajectory_and_save(xy_points, filename, x_bounds=None, y_bounds=None):
    # Extract X and Y coordinates from the list of points
    x_coords, y_coords = zip(*xy_points)

    # Plot the trajectory
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    for i in range(len(x_coords)):
        plt.plot(x_coords[i], y_coords[i], marker='o', linestyle='-', linewidth=1, color='blue')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectory Plot')
    plt.grid(True)
    if x_bounds:
        plt.xlim(x_bounds)
    if y_bounds:
        plt.ylim(y_bounds)
    plt.savefig(filename)

    # Close the plot to release resources
    plt.close()

def plot_heatmap_and_save(xy_points, filename, x_bounds=None, y_bounds=None):
    # Extract X and Y coordinates from the list of points
    x_coords, y_coords = zip(*xy_points)

    # Set up bins for 2D histogram
    if x_bounds is None:
        x_bounds = (min(x_coords), max(x_coords))
    if y_bounds is None:
        y_bounds = (min(y_coords), max(y_coords))

    # Create a 2D histogram
    aspect_ratio = (x_bounds[1] - x_bounds[0]) / (y_bounds[1] - y_bounds[0])
    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=50, range=[x_bounds, y_bounds])
    plt.figure(figsize=(15, 10))  # Adjust figure size as needed
    plt.imshow(heatmap.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='plasma')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xticks(fontsize=10)  # Set font size for x-axis tick labels
    plt.yticks(fontsize=10) 
    
    plt.colorbar(orientation='horizontal', pad=0.1, label="Frequency")  # Adjust pad as needed
    plt.gca().set_aspect(aspect_ratio, adjustable='box')

    plt.savefig(filename)
    plt.close()

#From HARMONI:
def get_plane(p1, p2, p3, anchor):
    # https://www.geeksforgeeks.org/program-to-find-equation-of-a-plane-passing-through-3-points/
    """ 
    Returns the plane that passes through anchor, and is perpendicular to the 
    plane defined by the 3 input points.
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
#     eye_mid = (reye+leye)/2
    x0, y0, z0 = anchor
    d = (- a * x0 - b * y0 - c * z0)
    return a, b, c, d

def is_visible(joints, other_joints, max_distance=1.8, fov=120):
    reye, leye, rear, lear, nose, neck = joints[15], joints[16], joints[17], joints[18], joints[0], joints[1]
    eye_mid = (reye+leye)/2
    a, b, c, _ = get_plane(rear, lear, neck, anchor=eye_mid)
    view_direction = np.array([a, b, c]) / np.linalg.norm(np.array([a, b, c]))
    if not is_in_cone(eye_mid, view_direction, nose, max_distance, fov)[0]:
        view_direction = -view_direction
    # make sure nose prediction is valid
    assert is_in_cone(eye_mid, view_direction, nose, max_distance, fov)[0]
    # check if each head joint of the other person in in cone of visibility
    for i in [0, 15, 16, 17, 18]: # only check the other person's head joints
        flag, angle, distance = is_in_cone(eye_mid, view_direction, other_joints[i], max_distance, fov)
        if flag:
            return True, angle, distance
    return False, angle, distance  # angle of the nose

def is_in_cone(start_point, direction, other_point, max_distance, fov):
    """ whether the other point is in the visible cone that's defined by start_point and direction."""
    other_point = Point(other_point)
    start_point = Point(start_point)
    angle = Vector(direction).angle_between(other_point-start_point)
    distance = other_point.distance_point(start_point) 
    if np.degrees(angle) < fov/2. and distance < max_distance:
        return True, np.degrees(angle), distance
    return False, np.degrees(angle), distance

def euclidean_distance_3d(point1, point2):
    distance = math.sqrt((point1[0] - point2[0])**2 + 
                         (point1[1] - point2[1])**2 +
                         (point1[2] - point2[2])**2)
    return distance

def euclidean_distance(point1, point2):
    distance = math.sqrt((point1[0] - point2[0])**2 + 
                         (point1[1] - point2[1])**2)
    return distance

def cumulative_sum(numbers):
    cumulative_sums = []
    current_sum = 0

    for num in numbers:
        current_sum += num
        cumulative_sums.append(current_sum)

    return cumulative_sums

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def plot_3d_trajectory(points_list, save_filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Flow Pattern')

    pos = np.array(points_list)
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color='red')

    plt.savefig(save_filename)
    plt.close()

def plot_3d_animation_traced(points_list, save_filename=None, interval=200):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    lines = [ax.plot([], [], [], color='gray', alpha=0.5)[0] for _ in range(len(points_list))]
    points = ax.scatter([], [], [], c='b', marker='o')

    def init():
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        points.set_offsets([])
        return points, *lines

    def update(frame):
        xs = [point[0] for point in points_list[frame]]
        ys = [point[1] for point in points_list[frame]]
        zs = [point[2] for point in points_list[frame]]

        points.set_offsets(np.c_[xs, ys])
        points._offsets3d = (xs, ys, zs)

        for i in range(frame):
            line_xs = [point[0] for point in points_list[i]]
            line_ys = [point[1] for point in points_list[i]]
            line_zs = [point[2] for point in points_list[i]]
            lines[i].set_data(line_xs, line_ys)
            lines[i].set_3d_properties(line_zs)

        ax.set_title(f'Time Step: {frame + 1}')
        return points, *lines

    anim = FuncAnimation(fig, update, frames=len(points_list), init_func=init, blit=True, interval=interval)
    anim.save(save_filename, writer='ffmpeg')

import math 
import numpy as np
import matplotlib.pyplot as plt

def plot_superimposed_bar_line_save(distance_list, cumulative_distance_list, save_filename, 
                                    include_cum=True, main_color="tab:blue", dist_limit=0.01, cumu_limit=0.3):
    time_steps = np.arange(1, len(distance_list) + 1)
    
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Distance Change', color=mcolors.CSS4_COLORS['black'])
    ax1.bar(time_steps, distance_list, color=main_color, alpha=0.5)
    ax1.tick_params(axis='y', labelcolor=mcolors.CSS4_COLORS['black'])
    ax1.set_ylim(0, dist_limit)

    if include_cum:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Cumulative Distance', color='tab:orange')
        ax2.plot(time_steps, cumulative_distance_list, color='tab:orange', linewidth=2)
        ax2.fill_between(time_steps, 0, cumulative_distance_list, color='tab:orange', alpha=0.2)
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.set_ylim(0, cumu_limit)

    fig.tight_layout()
    plt.savefig(save_filename)
    plt.close()

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def save_bar_chart(values, save_filename, title, ylabel, ymin, ylim, xlabel="Time Step"):
    plt.bar(range(len(values)), values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ymin, ylim) 
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_filename)
    plt.close()

import matplotlib.pyplot as plt
import seaborn as sns

def save_1d_state_plot(boolean_list, filename, fontsize=12):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow([boolean_list], cmap='plasma', aspect='auto')  # Use 'Reds' colormap for red style

    # Set labels and ticks
    ax.set_xlabel('Frame')
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xticks([])

    # Save the plot to a file
    plt.savefig(filename)
    plt.clf()
    plt.close()  # Close the plot to release resources

def save_3d_vectors_plot(vectors, origins, save_filename, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a colormap
    cmap = plt.cm.GnBu

    # Create a ScalarMappable object to map indices to colors
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=len(vectors) - 1))

    # Generate colors using the colormap
    colors = [sm.to_rgba(i) for i in range(len(vectors) + 100)]

    #Plotting 
    counter = 0 
    for vector, origin in zip(vectors, origins):
        counter += 1 
        ax.add_line(Line3D(*zip(origin, vector), linewidth=3, color=colors[counter]))

    max_coordinate = max(max(v) for v in vectors)
    min_coordinate = min(min(v) for v in vectors)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title is not None:
        ax.set_title(title)
    
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.tick_params(axis='both', which='major', labelsize=8)

    plt.savefig(save_filename)
    plt.clf()
    plt.close()

# Helper function to calculate angle between two 3D points
def angle_between_joints_3d(point1, point2, center_point):
    vector1 = (point1[0] - center_point[0], point1[1] - center_point[1], point1[2] - center_point[2])
    vector2 = (point2[0] - center_point[0], point2[1] - center_point[1], point2[2] - center_point[2])
    
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2 + vector1[2] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2 + vector2[2] ** 2)
    
    angle_radians = math.acos(dot_product / (magnitude1 * magnitude2))
    return angle_radians

def is_in_cone(start_point, direction, other_point, max_distance, fov):
    """ whether the other point is in the visible cone that's defined by start_point and direction."""
    other_point = Point(other_point)
    start_point = Point(start_point)
    angle = Vector(direction).angle_between(other_point-start_point)
    distance = other_point.distance_point(start_point) 
    if np.degrees(angle) < fov/2. and distance < max_distance:
        return True, np.degrees(angle), distance
    return False, np.degrees(angle), distance

# ________________________________________________________________________________________________


#Tracker movements - plot trajectory 
def tracklet_trajectories(positions, tracklet_index, sort_index):
    static_save_path = os.path.join(DS_EXP_PATH, "trajectory", str(sort_index).zfill(3) + "_" + str(tracklet_index).zfill(3) + ".png")
    plot_3d_trajectory(positions, static_save_path)

#Distances 
def tracklet_distance(positions, tracklet_index, sort_index): 
    distances = [] 
    last_position = positions[0]
    static_save_path = os.path.join(DS_EXP_PATH, "tracklet_dist", str(sort_index).zfill(3) + "_" + str(tracklet_index).zfill(3) + ".png") 
    for i, curr_position in enumerate(positions[1:]):
        curr_distance =  euclidean_distance(last_position, curr_position)
        distances.append(curr_distance)
        last_position = curr_position #update last_position 
    
    cumu_sum = cumulative_sum(distances) 
    plot_superimposed_bar_line_save(distances, cumu_sum, static_save_path, dist_limit=0.01)

from mpl_toolkits.mplot3d.art3d import Line3D

#Archived 
def tracklet_vision(vision_joints, tracklet_index, sort_index, max_distance=1.8, fov=120):
    vision_angles = []
    simlarity_stats = [] 
    origins = []
    last_direction = np.array([]) 
    for vision in vision_joints:
        reye, leye, rear, lear, nose, neck = vision
        eye_mid = (np.array(reye) + np.array(leye))/2
        a, b, c, _ = get_plane(np.array(rear), np.array(lear), np.array(neck), anchor=eye_mid)
        view_direction = np.array([a, b, c]) / np.linalg.norm(np.array([a, b, c]))
        if not is_in_cone(eye_mid, view_direction, nose, max_distance, fov)[0]:
                view_direction = -view_direction
        
        #Similarity 
        if last_direction.shape[0] > 0:
            cos_sim = cosine_similarity(view_direction, last_direction)
            simlarity_stats.append(cos_sim) 
        
        #Add to vision angles 
        angle = Vector(view_direction).angle_between(np.array(nose) - np.array(eye_mid))

        origins.append(eye_mid.tolist())
        vision_angles.append(view_direction) 
        last_direction = view_direction  

    #Plot similarity stats
    sim_stats_path = os.path.join(DS_EXP_PATH, "vision_sim", str(sort_index).zfill(3) + "_" + str(tracklet_index).zfill(3) + ".png")
    vectors_path = os.path.join(DS_EXP_PATH, "vision_vector", str(sort_index).zfill(3) + "_" + str(tracklet_index).zfill(3) + ".png")
    save_bar_chart(simlarity_stats, sim_stats_path, "Viewing Angle Similarity Progression", "Viewing Angle Cosine Similarity", ymin=-0.4, ylim=1.0, xlabel="Time Step")
    save_3d_vectors_plot(vision_angles, origins, vectors_path, "Scattered 3D Viewing Angles")

def tracklet_arms(arm_joints, tracklet_index, sort_index):
    elbow_angles = [] 
    elbow_distances = [] 
    wrist_distances = [] 
    for arm_joint_set in arm_joints:
        left_shoulder, left_elbow, left_wrist, right_shoulder, right_elbow, right_wrist = arm_joint_set 
        right_elbow_angle = angle_between_joints_3d(right_elbow, right_wrist, right_shoulder) 
        left_elbow_angle = angle_between_joints_3d(left_elbow, left_wrist, left_shoulder) 
        average_elbow_angle = (right_elbow_angle + left_elbow_angle) / 2 
        wrist_distance = euclidean_distance(left_wrist, right_wrist)
        elbow_distance = euclidean_distance(left_elbow, right_elbow)

        #Add to lists 
        elbow_angles.append(average_elbow_angle)
        elbow_distances.append(elbow_distance)
        wrist_distances.append(wrist_distance)

    #Create cumu sums 
    cum_elbow_dist = cumulative_sum(elbow_distances)
    cum_wrist_dist = cumulative_sum(wrist_distances)

    #Plot and save paths 
    elbow_dist_path = os.path.join(DS_EXP_PATH, "elbow_dist", str(sort_index).zfill(3) + "_" + str(tracklet_index).zfill(3) + ".png")
    elbow_angle_path = os.path.join(DS_EXP_PATH, "elbow_angle", str(sort_index).zfill(3) + "_" + str(tracklet_index).zfill(3) + ".png")
    wrist_dist_path = os.path.join(DS_EXP_PATH, "wrist_dist", str(sort_index).zfill(3) + "_" + str(tracklet_index).zfill(3) + ".png")
    plot_superimposed_bar_line_save(elbow_distances, cum_elbow_dist, elbow_dist_path, include_cum=False, main_color=mcolors.CSS4_COLORS['teal'], dist_limit=0.7)
    plot_superimposed_bar_line_save(wrist_distances, cum_wrist_dist, wrist_dist_path, include_cum=False, main_color=mcolors.CSS4_COLORS['springgreen'], dist_limit=1.0)
    save_bar_chart(elbow_angles, elbow_angle_path, "Average Elbow Angle Progression over Time", "Elbow Angle (Radians)", ymin=-0.0, ylim=1.0, xlabel="Time Step")
    
    
def calc_fingertip_distance(thumb, tips):
    dist_sum = sum([euclidean_distance(thumb, tip) for tip in tips])
    return dist_sum / len(tips) 

def tracklet_hands(finger_joints, tracklet_index, sort_index):
    tip_distances = [] 
    hand_distances = [] 
    for finger_set in finger_joints:
        #Hand distance 
        left_thumb, left_index, left_middle, left_pinky, left_ring = finger_set[:5] 
        right_thumb, right_index, right_middle, right_pinky, right_ring = finger_set[5:] 
        hand_distance = euclidean_distance(left_thumb, right_thumb)
        hand_distances.append(hand_distance)
        
        #Tip Distance 
        left_tip_dist = calc_fingertip_distance(left_thumb, [left_index, left_middle, left_pinky, left_ring])
        right_tip_dist = calc_fingertip_distance(right_thumb, [right_index, right_middle, right_pinky, right_ring])
        tip_distances.append((left_tip_dist + right_tip_dist) / 2)

    #Create cumu sums 
    cumu_tip_dist = cumulative_sum(tip_distances)
    cumu_hand_dist = cumulative_sum(hand_distances)

    #Plotting 
    tips_dist_path = os.path.join(DS_EXP_PATH, "tip_dist", str(sort_index).zfill(3) + "_" + str(tracklet_index).zfill(3) + ".png")
    hands_dist_path = os.path.join(DS_EXP_PATH, "hand_dist", str(sort_index).zfill(3) + "_" + str(tracklet_index).zfill(3) + ".png")
    plot_superimposed_bar_line_save(tip_distances, cumu_tip_dist, tips_dist_path, include_cum=False, main_color=mcolors.CSS4_COLORS['royalblue'], dist_limit=0.12)
    plot_superimposed_bar_line_save(hand_distances, cumu_hand_dist, hands_dist_path, include_cum=False, main_color=mcolors.CSS4_COLORS['mediumslateblue'], dist_limit=1.0)

#Template Function for Analyzing Joints 
def joint_analysis_wrapper(name="c1_v2"):
    #Data structures 
    joint_metadata_path = "/pasteur/u/bencliu/baseline/data/datasets/experiments/crowd/joint_out"
    joint_metadata_path = os.path.join(joint_metadata_path, name)
    tracklets_core = {} 
    tracklets = set() 

    #Create save paths 
    tracklets_core_save_path = "/pasteur/u/bencliu/baseline/data/datasets/experiments/downstream_subset"
    tracklets_core_save_path = os.path.join(tracklets_core_save_path, name + "_tracklets_core.pkl")

    for i in range(6800, 9500): #Work with smaller subset to begin with 
        img_suffix = str(i).zfill(6) + ".pkl"
        metadata_path = os.path.join(joint_metadata_path, img_suffix) #Working with particular frame 
        try:
            metadata = read_pickle(metadata_path) 
        except:
            print(metadata_path, " does not exist.")
            continue 
        joints = metadata['joints_3d'] 
        trackers = metadata['trackers'] 
        #Process metadata and index necessary joints as 3D values
        for joint_set, tracker in zip(joints, trackers):
            vision_joints = [joint_set[56], joint_set[57], joint_set[58], joint_set[59], joint_set[55], joint_set[12]]
            arm_joints = [joint_set[16], joint_set[18], joint_set[20], joint_set[17], joint_set[19], joint_set[21]] #shoulder, elbow, wrist [LR] 
            finger_joints = [joint_set[39], joint_set[27], joint_set[30], joint_set[33], joint_set[36], joint_set[54], joint_set[42], joint_set[45], joint_set[48], joint_set[51]] #Thumb, other finger tips (LR) 
            spine_joint = joint_set[6] 

            if tracker in tracklets:
                tracklets_core[tracker]['positions'].append(spine_joint)
                tracklets_core[tracker]['vision'].append(vision_joints)
                tracklets_core[tracker]['arms'].append(arm_joints)
                tracklets_core[tracker]['fingers'].append(finger_joints)
            else:
                tracklets.add(tracker) 
                tracklets_core[tracker] = {
                    "positions" : [spine_joint],
                    "vision" : [vision_joints], 
                    "arms" : [arm_joints],
                    "fingers" : [finger_joints] 
                }
        
        if i % 500 == 0:
            print("Completed: ", i)

    write_pickle(tracklets_core, tracklets_core_save_path)
    sorted_keys = sorted(tracklets_core, key=lambda k: len(tracklets_core[k]['positions']), reverse=True)
    main_keys_c4 = [1, 15, 22, 3, 20, 10]
    main_keys_c1 = [] 

def is_visible(joints, other_joints, max_distance=1.8, fov=120):
    reye, leye, rear, lear, nose, neck = joints[15], joints[16], joints[17], joints[18], joints[0], joints[1]
    eye_mid = (reye+leye)/2
    a, b, c, _ = get_plane(rear, lear, neck, anchor=eye_mid)
    view_direction = np.array([a, b, c]) / np.linalg.norm(np.array([a, b, c]))
    if not is_in_cone(eye_mid, view_direction, nose, max_distance, fov)[0]:
        view_direction = -view_direction
    # make sure nose prediction is valid
    assert is_in_cone(eye_mid, view_direction, nose, max_distance, fov)[0]
    # check if each head joint of the other person in in cone of visibility
    for i in [0, 15, 16, 17, 18]: # only check the other person's head joints
        flag, angle, distance = is_in_cone(eye_mid, view_direction, other_joints[i], max_distance, fov)
        if flag:
            return True, angle, distance
    return False, angle, distance  # angle of the nose

def is_in_cone(start_point, direction, other_point, max_distance, fov):
    """ whether the other point is in the visible cone that's defined by start_point and direction."""
    other_point = Point(other_point)
    start_point = Point(start_point)
    angle = Vector(direction).angle_between(other_point-start_point)
    distance = other_point.distance_point(start_point) 
    if np.degrees(angle) < fov/2. and distance < max_distance:
        return True, np.degrees(angle), distance
    return False, np.degrees(angle), distance

def compute_mutual_distance_attention(name): 
    joint_metadata_path = "/pasteur/u/bencliu/baseline/data/datasets/experiments/crowd/joint_out"
    joint_metadata_path = os.path.join(joint_metadata_path, name)
    for i in range(1, 9000): #Work with smaller subset to begin with 
        img_suffix = str(i).zfill(6) + ".pkl"
        metadata_path = os.path.join(joint_metadata_path, img_suffix) 
        try:
            metadata = read_pickle(metadata_path) 
        except:
            print(metadata_path, " does not exist.")
            continue 
        joints = metadata['joints_3d'] 
        trackers = metadata['trackers'] 
        #Process metadata and index necessary joints as 3D values
        for joint_set, tracker in zip(joints, trackers):
            pass 
    
if __name__ == "__main__": 
    print("Starting experimentation demo")
    geometry_analysis_high_wrapper() 
    breakpoint() 
    #joint_analysis_wrapper(name="c4_v3")
    #analysis_wrapper(name="c4_v3")

