import numpy as np
import pickle
import os
import torch
import io
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from empirical.util import read_pickle, apply_transform

def process_files(folder_path, debug=False):
    paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pkl')])
    if debug:
        paths = paths[:100]
    data = []
    for path in paths:
        f = read_pickle(path)
        joints, vertices = apply_transform(f, False)
        one_frame = [joints[i, 0, (0, 2)] for i in range(joints.shape[0])]
        one_frame = np.array(one_frame)
        data.append(one_frame)
    return np.array(data, dtype=object)

def calculate_distance(data):
    kmeans = KMeans(n_clusters=1, random_state=0, n_init=10) #Cluster all points into one cluster
    centroid_dispersion = [] #Output is a list, where each element represents the average distance of the group from the centroid for each time
    prev_centroid = None 
    centroid_drift = [0] 
    for frame in data:
        kmeans.fit(frame)
        centroid = kmeans.cluster_centers_[0]
        if prev_centroid is not None:
            centroid_dist = euclidean(prev_centroid, centroid)
            centroid_drift.append(distance)
        prev_centroid = centroid
        distance = 0 
        for point in frame: #For each pelvis location, calculate its distance from the centroid
            distance += euclidean(point, centroid)
        distance = distance / len(frame) #Then take the average, and append to output list
        centroid_dispersion.append(distance)
    return centroid_dispersion, centroid_drift 

def pipeline_tester(folder_path="/pasteur/data/ghent_surg/full_hmr_outputs/220704_22022/joint_out"):
    data = process_files(folder_path, debug=True)
    centroid_dispersion, centroid_drift = calculate_distance(data)

if __name__ == "__main__":
    pipeline_tester()