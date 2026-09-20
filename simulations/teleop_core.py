import numpy as np
import mediapipe as mp

# MediaPipe Tasks setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='assets/models/hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)
landmarker = HandLandmarker.create_from_options(options)

def angle_between(p1, midpt, p2, plane=np.array([1, 1, 1])):
    """Computes the angle between two 3d points and a midpoint"""
    ba = (p1 - midpt) * plane
    bc = (p2 - midpt) * plane
    # Protect against division by zero
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # Clip to avoid numerical issues with arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def analyze_hand_landmarks(landmarks):
    """Analyze the hand landmarks and return the 24 joint angles as used by DexHand."""
    # Convert to numpy array
    num_landmarks = len(landmarks)
    joint_xyz = np.zeros((num_landmarks, 3))
    for i in range(num_landmarks):
        joint_xyz[i] = np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z])

    joint_angles = np.zeros(24)

    # First finger, index
    joint_angles[0] = 180 - angle_between(joint_xyz[0], joint_xyz[5], joint_xyz[6]) - 10 # pitch
    joint_angles[1] = 90 - angle_between(joint_xyz[9], joint_xyz[5], joint_xyz[6])       # yaw
    joint_angles[2] = 180 - angle_between(joint_xyz[5], joint_xyz[6], joint_xyz[7])      # knuckle
    joint_angles[3] = 180 - angle_between(joint_xyz[6], joint_xyz[7], joint_xyz[8])      # tip

    # Second finger, middle
    joint_angles[4] = 180 - angle_between(joint_xyz[0], joint_xyz[9], joint_xyz[10])
    joint_angles[5] = angle_between(joint_xyz[5], joint_xyz[9], joint_xyz[10]) - 90 - 15
    joint_angles[6] = 180 - angle_between(joint_xyz[9], joint_xyz[10], joint_xyz[11])
    joint_angles[7] = 180 - angle_between(joint_xyz[10], joint_xyz[11], joint_xyz[12])

    # Third finger, ring
    joint_angles[8] = 180 - angle_between(joint_xyz[0], joint_xyz[13], joint_xyz[14])
    joint_angles[9] = angle_between(joint_xyz[9], joint_xyz[13], joint_xyz[14]) - 90
    joint_angles[10] = 180 - angle_between(joint_xyz[13], joint_xyz[14], joint_xyz[15])
    joint_angles[11] = 180 - angle_between(joint_xyz[14], joint_xyz[15], joint_xyz[16])

    # Fourth finger, pinky
    joint_angles[12] = 180 - angle_between(joint_xyz[0], joint_xyz[17], joint_xyz[18])
    joint_angles[13] = angle_between(joint_xyz[13], joint_xyz[17], joint_xyz[18]) - 90
    joint_angles[14] = 180 - angle_between(joint_xyz[17], joint_xyz[18], joint_xyz[19])
    joint_angles[15] = 180 - angle_between(joint_xyz[18], joint_xyz[19], joint_xyz[20])

    # Thumb, upper, lower, and flexion
    joint_angles[16] = 180 - angle_between(joint_xyz[1], joint_xyz[2], joint_xyz[4]) # pitch
    joint_angles[17] = 60 - angle_between(joint_xyz[2], joint_xyz[1], joint_xyz[5])  # roll
    joint_angles[18] = 180 - angle_between(joint_xyz[2], joint_xyz[3], joint_xyz[4]) # knuckle
    # Tip (approximate from CMC-MCP-IP)
    joint_angles[19] = 180 - angle_between(joint_xyz[1], joint_xyz[3], joint_xyz[4]) 
    # yaw (approximate)
    joint_angles[20] = 0.0

    # Wrist pitch and yaw from global frame
    v = joint_xyz[13] - joint_xyz[0]
    pitch_angle = np.degrees(np.arctan2(v[2] * 2.5, -v[1])) # Z vs UP
    yaw_angle = np.degrees(np.arctan2(v[0], -v[1]))         # X vs UP
    
    joint_angles[21] = pitch_angle
    joint_angles[22] = -yaw_angle  # Flipped yaw for natural mirroring
    joint_angles[23] = pitch_angle

    return joint_angles
