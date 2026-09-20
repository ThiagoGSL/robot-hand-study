import cv2
import time
import numpy as np
import mujoco
import mujoco.viewer
import mediapipe as mp
from teleop_core import landmarker, analyze_hand_landmarks

def main():
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path("scenes/scene_horizontal_drop.xml")
    data = mujoco.MjData(model)

    # Position the hand dynamically
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_root")
    if hand_body_id != -1:
        jnt_id = model.body_jntadr[hand_body_id]
        qpos_adr = model.jnt_qposadr[jnt_id]
        
        data.qpos[qpos_adr+0] = 0.06    # X
        data.qpos[qpos_adr+1] = -0.31   # Y
        data.qpos[qpos_adr+2] = 0.32    # Z
        data.qpos[qpos_adr+3:qpos_adr+7] = [-0.5, 0.5, -0.5, -0.5] # Quat

    # Get Mocap ID for the supports
    supports_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "supports")
    mocap_id = model.body_mocapid[supports_body_id]

    # Pre-compute actuator IDs for fast lookup
    actuator_names = [
        "act_index_pitch", "act_index_yaw", "act_index_knuckle", "act_index_tip",
        "act_middle_pitch", "act_middle_yaw", "act_middle_knuckle", "act_middle_tip",
        "act_ring_pitch", "act_ring_yaw", "act_ring_knuckle", "act_ring_tip",
        "act_pinky_pitch", "act_pinky_yaw", "act_pinky_knuckle", "act_pinky_tip",
        "act_thumb_pitch", "act_thumb_roll", "act_thumb_knuckle", "act_thumb_tip", "act_thumb_yaw",
        "act_wrist_pitch_lower", "act_wrist_yaw", "act_wrist_pitch_upper"
    ]
    
    actuator_ids = []
    for name in actuator_names:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx == -1:
            print(f"Warning: Actuator {name} not found in model!")
        actuator_ids.append(idx)

    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Launching Passive Viewer...")
    
    # Force the hand to start wide open (0 angle for all joints) before tracking kicks in
    for idx in actuator_ids:
        if idx != -1:
            data.ctrl[idx] = 0.0
    # Reset all joint positions (fingers) to 0.
    if hand_body_id != -1:
        jnt_id = model.body_jntadr[hand_body_id]
        qpos_adr = model.jnt_qposadr[jnt_id]
        data.qpos[qpos_adr+7:] = 0.0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running() and cap.isOpened():
            # Step physics to match real wall-clock time
            time_since_start = time.time() - start_time
            while data.time < time_since_start:
                # FREEZE THE FOREARM
                if hand_body_id != -1:
                    jnt_id = model.body_jntadr[hand_body_id]
                    qpos_adr = model.jnt_qposadr[jnt_id]
                    dof_adr = model.jnt_dofadr[jnt_id]
                    
                    data.qpos[qpos_adr+0] = 0.06
                    data.qpos[qpos_adr+1] = -0.31
                    data.qpos[qpos_adr+2] = 0.32
                    data.qpos[qpos_adr+3:qpos_adr+7] = [-0.5, 0.5, -0.5, -0.5]
                    data.qvel[dof_adr:dof_adr+6] = 0.0
                
                mujoco.mj_step(model, data)
            
            # Read webcam
            success, image = cap.read()
            if not success:
                break
            
            # Mirror image for natural teleop
            image = cv2.flip(image, 1)
            # Convert BGR to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            results = landmarker.detect(mp_image)

            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    # Draw points on the hand so the user can see tracking is working
                    for lm in hand_landmarks:
                        cx, cy = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                        cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
                        
                    # Calculate angles
                    angles = analyze_hand_landmarks(hand_landmarks)
                    
                    # MediaPipe 3D yaw (finger spread) is extremely noisy from a single webcam.
                    # We zero out the yaw for the 4 main fingers so they don't wildly cross each other.
                    angles[1] = 0
                    angles[5] = 0
                    angles[9] = 0
                    angles[13] = 0
                    
                    # Apply to MuJoCo
                    for i, act_idx in enumerate(actuator_ids):
                        if act_idx != -1:
                            # Convert to radians
                            rad_val = np.radians(angles[i])
                            
                            # Grab actuator ctrlrange
                            ctrl_min = model.actuator_ctrlrange[act_idx][0]
                            ctrl_max = model.actuator_ctrlrange[act_idx][1]
                            
                            # Clamp value
                            clamped_val = np.clip(rad_val, ctrl_min, ctrl_max)
                            
                            # Apply smooth filter to reduce jitter 
                            # Fingers can be fast (0.3), but the wrist is heavy and should be smoother (0.05) to prevent whipping
                            alpha = 0.05 if 'wrist' in actuator_names[i] else 0.3
                            data.ctrl[act_idx] = (alpha * clamped_val) + ((1 - alpha) * data.ctrl[act_idx])

            # Sync viewer with new physics state
            viewer.sync()
            
            # Display OpenCV window
            cv2.imshow("DexHand Teleop", image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): # ESC or Q to quit
                break
            elif key == ord('d'): # D to Drop
                print("Derrubando os suportes da barra!")
                if mocap_id != -1:
                    data.mocap_pos[mocap_id][2] = -5.0 # Drop supports 5 meters down
            elif key == ord('r'): # R to Reset
                print("Resetando a física e a posição da mão...")
                mujoco.mj_resetData(model, data)
                if mocap_id != -1:
                    data.mocap_pos[mocap_id][2] = 0.27
                if hand_body_id != -1:
                    jnt_id = model.body_jntadr[hand_body_id]
                    qpos_adr = model.jnt_qposadr[jnt_id]
                    data.qpos[qpos_adr+0] = 0.06
                    data.qpos[qpos_adr+1] = -0.31
                    data.qpos[qpos_adr+2] = 0.32
                    data.qpos[qpos_adr+3:qpos_adr+7] = [-0.5, 0.5, -0.5, -0.5]
                    data.qpos[qpos_adr+7:] = 0.0
                start_time = time.time()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
