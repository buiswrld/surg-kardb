MAIN_JOINTS = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
]

# #Updated categories 
# pelvic_joints = ['pelvis', 'left_hip','right_hip']
# arm_joints = ['left_elbow','right_elbow','left_wrist','right_wrist']
# head_joints = ["head",'jaw','nose','right_eye','left_eye','right_ear','left_ear']
# thorax_joints = ['left_shoulder','right_shoulder', 'left_collar', 'right_collar', 'neck'] 
# leg_joints = ['left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot']
# spine_joints = ['spine1', 'spine2', 'spine3',]

# core_joints = ['pelvis', 'left_hip','right_hip']
# spine_joints = ['spine1', 'spine2', 'spine3',]
# head_joints = ["head",'jaw','nose','right_eye','left_eye','right_ear','left_ear']
# leg_joints = ['left_knee','right_knee','spine2','left_ankle','right_ankle','spine3','left_foot','right_foot'] 
# arm_joints = ['left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist']

SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]


#https://github.com/vchoutas/smplx/issues/14 -- first 127 joints 
JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye_smplhf',
    'right_eye_smplhf',
    'left_index1',
    'left_index2',
    'left_index3',
    'left_middle1',
    'left_middle2',
    'left_middle3',
    'left_pinky1',
    'left_pinky2',
    'left_pinky3',
    'left_ring1',
    'left_ring2',
    'left_ring3',
    'left_thumb1',
    'left_thumb2',
    'left_thumb3',
    'right_index1',
    'right_index2',
    'right_index3',
    'right_middle1',
    'right_middle2',
    'right_middle3',
    'right_pinky1',
    'right_pinky2',
    'right_pinky3',
    'right_ring1',
    'right_ring2',
    'right_ring3',
    'right_thumb1',
    'right_thumb2',
    'right_thumb3',
    'nose',
    'right_eye',
    'left_eye',
    'right_ear',
    'left_ear',
    'left_big_toe',
    'left_small_toe',
    'left_heel',
    'right_big_toe',
    'right_small_toe',
    'right_heel',
    'left_thumb',
    'left_index',
    'left_middle',
    'left_ring',
    'left_pinky',
    'right_thumb',
    'right_index',
    'right_middle',
    'right_ring',
    'right_pinky',
    'right_eye_brow1',
    'right_eye_brow2',
    'right_eye_brow3',
    'right_eye_brow4',
    'right_eye_brow5',
    'left_eye_brow5',
    'left_eye_brow4',
    'left_eye_brow3',
    'left_eye_brow2',
    'left_eye_brow1',
    'nose1',
    'nose2',
    'nose3',
    'nose4',
    'right_nose_2',
    'right_nose_1',
    'nose_middle',
    'left_nose_1',
    'left_nose_2',
    'right_eye1',
    'right_eye2',
    'right_eye3',
    'right_eye4',
    'right_eye5',
    'right_eye6',
    'left_eye4',
    'left_eye3',
    'left_eye2',
    'left_eye1',
    'left_eye6',
    'left_eye5',
    'right_mouth_1',
    'right_mouth_2',
    'right_mouth_3',
    'mouth_top',
    'left_mouth_3',
    'left_mouth_2',
    'left_mouth_1',
    'left_mouth_5',  # 59 in OpenPose output
    'left_mouth_4',  # 58 in OpenPose output
    'mouth_bottom',
    'right_mouth_4',
    'right_mouth_5',
    'right_lip_1',
    'right_lip_2',
    'lip_top',
    'left_lip_2',
    'left_lip_1',
    'left_lip_3',
    'lip_bottom',
    'right_lip_3',
    # Face contour
    'right_contour_1',
    'right_contour_2',
    'right_contour_3',
    'right_contour_4',
    'right_contour_5',
    'right_contour_6',
    'right_contour_7',
    'right_contour_8',
    'contour_middle',
    'left_contour_8',
    'left_contour_7',
    'left_contour_6',
    'left_contour_5',
    'left_contour_4',
    'left_contour_3',
    'left_contour_2',
    'left_contour_1',
]

# POSE ABLATION 
pelvic_pose = ['pelvis', 'left_hip','right_hip']
arm_pose = ['left_elbow','right_elbow','left_wrist','right_wrist']
head_pose = ["head"]
thorax_pose = ['left_shoulder','right_shoulder', 'left_collar', 'right_collar', 'neck'] 
leg_pose = ['left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot']
spine_pose = ['spine1', 'spine2', 'spine3',]
left_side_joints = ['left_hip', 'left_knee', 'left_ankle', 'left_foot',
    'left_collar', 'left_shoulder', 'left_elbow', 'left_wrist',
    'left_eye', 'left_ear']
right_side_joints = ['right_hip', 'right_knee', 'right_ankle', 'right_foot',
    'right_collar', 'right_shoulder', 'right_elbow', 'right_wrist',
    'right_eye', 'right_ear']
upper_body_joints = ['spine1', 'spine2', 'spine3', 'neck', 'head', 'jaw', 'nose',
    'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_collar', 'right_collar', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
lower_body_joints = ['pelvis', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    'left_foot', 'right_foot']
knee_joints = ['left_knee', 'right_knee']
ankle_joints = ['left_ankle', 'right_ankle']
foot_joints = ['left_foot', 'right_foot']
kiran_pose = ["left_eye", "left_shoulder", "right_knee", "right_ankle"]
anti_kiran_pose = ["right_eye", "right_shoulder", "left_knee", "left_ankle"]
left_arm_joints = ['left_shoulder', 'left_elbow', 'left_wrist']
right_arm_joints = ['right_shoulder', 'right_elbow', 'right_wrist']
left_leg_joints = ['left_hip', 'left_knee', 'left_ankle', 'left_foot']
right_leg_joints = ['right_hip', 'right_knee', 'right_ankle', 'right_foot']
hip_joints = ['left_hip', 'right_hip']


# Indices for the joints
left_side_indices = [JOINT_NAMES.index(j) for j in left_side_joints if j in JOINT_NAMES]
lower_body_indices = [JOINT_NAMES.index(j) for j in lower_body_joints if j in JOINT_NAMES]
upper_body_indices = [JOINT_NAMES.index(j) for j in upper_body_joints if j in JOINT_NAMES]
right_side_indices = [JOINT_NAMES.index(j) for j in right_side_joints if j in JOINT_NAMES]
pelvic_indices_pose = [SMPL_JOINT_NAMES.index(joint) for joint in pelvic_pose]
arm_indices_pose = [SMPL_JOINT_NAMES.index(joint) for joint in arm_pose]
head_indices_pose = [SMPL_JOINT_NAMES.index(joint) for joint in head_pose]
thorax_indices_pose = [SMPL_JOINT_NAMES.index(joint) for joint in thorax_pose]
leg_indices_pose = [SMPL_JOINT_NAMES.index(joint) for joint in leg_pose]
spine_indices_pose = [SMPL_JOINT_NAMES.index(joint) for joint in spine_pose]
knee_indices = [JOINT_NAMES.index(j) for j in knee_joints if j in JOINT_NAMES]
ankle_indices = [JOINT_NAMES.index(j) for j in ankle_joints if j in JOINT_NAMES]
foot_indices = [JOINT_NAMES.index(j) for j in foot_joints if j in JOINT_NAMES]
kiran_indices = [JOINT_NAMES.index(j) for j in kiran_pose if j in JOINT_NAMES]
left_arm_indices = [JOINT_NAMES.index(j) for j in left_arm_joints if j in JOINT_NAMES]
right_arm_indices = [JOINT_NAMES.index(j) for j in right_arm_joints if j in JOINT_NAMES]
left_leg_indices = [JOINT_NAMES.index(j) for j in left_leg_joints if j in JOINT_NAMES]
right_leg_indices = [JOINT_NAMES.index(j) for j in right_leg_joints if j in JOINT_NAMES]
hip_indices = [JOINT_NAMES.index(j) for j in hip_joints if j in JOINT_NAMES]
anti_kiran_indices = [JOINT_NAMES.index(j) for j in anti_kiran_pose if j in JOINT_NAMES]


# JOINT CLASS ABLATION 
pelvic_joints = ['pelvis', 'left_hip','right_hip']
arm_joints = ['left_elbow','right_elbow','left_wrist','right_wrist']
head_joints = ["head",'jaw','nose','right_eye','left_eye','right_ear','left_ear']
thorax_joints = ['left_shoulder','right_shoulder', 'left_collar', 'right_collar', 'neck'] 
leg_joints = ['left_knee','right_knee','left_ankle','right_ankle','left_foot','right_foot']
spine_joints = ['spine1', 'spine2', 'spine3',]

pelvic_indices = [JOINT_NAMES.index(joint) for joint in pelvic_joints]
arm_indices = [JOINT_NAMES.index(joint) for joint in arm_joints]
head_indices = [JOINT_NAMES.index(joint) for joint in head_joints]
thorax_indices = [JOINT_NAMES.index(joint) for joint in thorax_joints]
leg_indices = [JOINT_NAMES.index(joint) for joint in leg_joints]
spine_indices = [JOINT_NAMES.index(joint) for joint in spine_joints]



# JOINT INDIVIDUAL ABLATION [Head/Hand Joints] 
head_struct_joints = ['nose', 'neck'] 
eye_joints = ['right_eye', 'left_eye']
ear_joints = ['right_ear', 'left_ear'] 
elbow_joints = ['left_elbow','right_elbow',] 
wrist_joints = ['left_wrist','right_wrist'] 

head_struct_indices = [JOINT_NAMES.index(joint) for joint in head_struct_joints]
eye_indices = [JOINT_NAMES.index(joint) for joint in eye_joints]
ear_indices = [JOINT_NAMES.index(joint) for joint in ear_joints]
elbow_indices = [JOINT_NAMES.index(joint) for joint in elbow_joints]
wrist_indices = [JOINT_NAMES.index(joint) for joint in wrist_joints]







# ARCHIVED 
# Categories
head_neck_joints = ['neck', 'head', 'left_collar', 'right_collar']
hand_arm_joints = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
abdominal_spinal_joints = ['pelvis', 'spine1', 'spine2', 'spine3', ]
lower_body_joints = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot"]
elbow_wrist_joints = ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'] #Extra 

# Index lists
lower_body_indices = [JOINT_NAMES.index(joint) for joint in lower_body_joints]
head_neck_indices = [JOINT_NAMES.index(joint) for joint in head_neck_joints]
hand_arm_indices = [JOINT_NAMES.index(joint) for joint in hand_arm_joints]
abdominal_spinal_joints_indices = [JOINT_NAMES.index(joint) for joint in abdominal_spinal_joints]
elbow_wrist_indices = [JOINT_NAMES.index(joint) for joint in elbow_wrist_joints]
general_joints_indices = lower_body_indices + head_neck_indices + hand_arm_indices + abdominal_spinal_joints_indices

