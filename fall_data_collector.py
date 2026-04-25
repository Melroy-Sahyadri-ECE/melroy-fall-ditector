"""
Fall Detection Data Collector
==============================
Uses the existing YOLO-Pose skeleton to collect labeled training data.

How to use:
  1. Run this script (camera opens with skeleton)
  2. Press 'f' then DO A FALL — records 2 seconds as "Fall"
  3. Press 'n' then DO NORMAL ACTIONS — records 2 seconds as "Normal" 
  4. Record 10-15 falls and 20-30 normal actions (stand, sit, bend, walk, lie down)
  5. Press 'q' to save and quit
  6. Then run: python train_fall_model.py

The collector captures sliding windows of skeleton features,
so the model learns the TEMPORAL PATTERN of a fall.
"""

import cv2
import numpy as np
import time
import math
import os
import json
import threading
from collections import deque
from ultralytics import YOLO

# Keypoint indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "fall_training_data")

WINDOW_SIZE = 15        # frames per window (~0.5 sec at 30fps)
RECORD_FRAMES = 60      # total frames to record per action (~2 sec)
CAPTURE_INTERVAL = 0.0  # no delay — capture every frame for temporal data


def _angle_3pts(ax, ay, bx, by, cx, cy):
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax * bcx + bay * bcy
    mag = math.sqrt(bax**2 + bay**2) * math.sqrt(bcx**2 + bcy**2)
    if mag < 1e-8:
        return 0.0
    return math.degrees(math.acos(max(-1, min(1, dot / mag))))


def _torso_angle(sh_cx, sh_cy, hp_cx, hp_cy):
    dx, dy = sh_cx - hp_cx, sh_cy - hp_cy
    if abs(dy) < 1e-6:
        return 90.0
    return abs(math.degrees(math.atan2(dx, dy)))


def extract_frame_features(keypoints, confs, frame_h, frame_w):
    """Extract per-frame features from skeleton keypoints.
    Returns 8 features or None if insufficient keypoints."""
    visible = confs > 0.3
    if visible.sum() < 6:
        return None

    def kp(idx):
        return keypoints[idx][0], keypoints[idx][1]

    nose_x, nose_y = kp(NOSE)
    l_sh_x, l_sh_y = kp(L_SHOULDER)
    r_sh_x, r_sh_y = kp(R_SHOULDER)
    l_hp_x, l_hp_y = kp(L_HIP)
    r_hp_x, r_hp_y = kp(R_HIP)
    l_kn_x, l_kn_y = kp(L_KNEE)
    r_kn_x, r_kn_y = kp(R_KNEE)
    l_an_x, l_an_y = kp(L_ANKLE)
    r_an_x, r_an_y = kp(R_ANKLE)

    sh_cx, sh_cy = (l_sh_x + r_sh_x) / 2, (l_sh_y + r_sh_y) / 2
    hp_cx, hp_cy = (l_hp_x + r_hp_x) / 2, (l_hp_y + r_hp_y) / 2

    torso_len = math.sqrt((sh_cx - hp_cx)**2 + (sh_cy - hp_cy)**2)
    if torso_len < 5:
        return None

    t_angle = _torso_angle(sh_cx, sh_cy, hp_cx, hp_cy)
    hip_y_norm = hp_cy / frame_h

    l_knee = _angle_3pts(l_hp_x, l_hp_y, l_kn_x, l_kn_y, l_an_x, l_an_y)
    r_knee = _angle_3pts(r_hp_x, r_hp_y, r_kn_x, r_kn_y, r_an_x, r_an_y)
    avg_knee = (l_knee + r_knee) / 2

    vis_xs = keypoints[visible, 0]
    vis_ys = keypoints[visible, 1]
    bbox_w = vis_xs.max() - vis_xs.min() if len(vis_xs) > 2 else 0
    bbox_h = vis_ys.max() - vis_ys.min() if len(vis_ys) > 2 else 1
    aspect_ratio = bbox_w / max(bbox_h, 1)

    sh_above = 1.0 if sh_cy < hp_cy else 0.0
    nose_below = 1.0 if (nose_y > hp_cy and confs[NOSE] > 0.3) else 0.0

    return [t_angle, hip_y_norm, avg_knee, aspect_ratio,
            sh_above, nose_below, torso_len / frame_h, bbox_h / frame_h]


FRAME_FEATURE_NAMES = [
    "torso_angle", "hip_y_norm", "avg_knee_angle", "aspect_ratio",
    "sh_above_hp", "nose_below_hips", "torso_len_norm", "body_height_norm"
]


def build_window_features(frame_buffer):
    """Build temporal features from a window of frame features.
    
    Takes WINDOW_SIZE frames of 8 features each and produces:
    - Start frame features (8)
    - End frame features (8)
    - Delta features (end - start) (8)
    - Max velocity per feature across window (8)
    - Statistics: mean, std of key features (6)
    Total: 38 features per window
    """
    if len(frame_buffer) < WINDOW_SIZE:
        return None

    window = list(frame_buffer)[-WINDOW_SIZE:]
    arr = np.array(window)  # [WINDOW_SIZE, 8]

    start = arr[0]
    end = arr[-1]
    delta = end - start

    # Frame-to-frame velocities
    velocities = np.diff(arr, axis=0)  # [WINDOW_SIZE-1, 8]
    max_vel = velocities.max(axis=0)

    # Stats on key signals (torso_angle=0, hip_y=1)
    torso_mean = arr[:, 0].mean()
    torso_std = arr[:, 0].std()
    hip_mean = arr[:, 1].mean()
    hip_std = arr[:, 1].std()
    knee_mean = arr[:, 2].mean()
    ar_mean = arr[:, 3].mean()

    features = np.concatenate([
        start, end, delta, max_vel,
        [torso_mean, torso_std, hip_mean, hip_std, knee_mean, ar_mean]
    ])
    return features.tolist()


WINDOW_FEATURE_COUNT = 38  # 8*4 + 6


def main():
    print("=" * 58)
    print("  FALL DETECTION — Training Data Collector")
    print("=" * 58)
    print()
    print("  Press 'f' → then FALL (records 2 sec as 'Fall')")
    print("  Press 'n' → then do NORMAL action (records 2 sec as 'Normal')")
    print("  Press 'q' → save and quit")
    print()
    print("  Record 10-15 falls + 20-30 normal actions for best results")
    print("  Normal = stand, sit, bend, walk, lie down slowly, wave arms")
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    print("[1/2] Loading YOLOv8n-pose...")
    model = YOLO("yolov8n-pose.pt")
    print("[2/2] Opening camera...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Collection state
    all_windows = {"Fall": [], "Normal": []}
    frame_buffer = deque(maxlen=60)
    recording = False
    record_label = ""
    record_count = 0
    record_start = 0

    print("[INFO] Camera ready. Press 'f' or 'n' to start recording.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Run pose
        results = model(frame, verbose=False, conf=0.4)

        # Extract features from first detected person
        features = None
        if results and results[0].keypoints is not None:
            kps_data = results[0].keypoints.data.cpu().numpy()
            if len(kps_data) > 0:
                kps = kps_data[0]
                keypoints = kps[:, :2]
                confs = kps[:, 2]
                features = extract_frame_features(keypoints, confs, h, w)

                # Draw skeleton
                for i in range(17):
                    if confs[i] > 0.3:
                        x, y = int(keypoints[i][0]), int(keypoints[i][1])
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Buffer features
        if features is not None:
            frame_buffer.append(features)

        # Recording
        if recording:
            record_count += 1
            # Extract sliding windows during recording
            if len(frame_buffer) >= WINDOW_SIZE and features is not None:
                win_features = build_window_features(frame_buffer)
                if win_features is not None:
                    all_windows[record_label].append(win_features)

            # Progress
            pct = min(100, record_count * 100 // RECORD_FRAMES)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            color = (0, 0, 255) if record_label == "Fall" else (0, 200, 0)
            cv2.putText(frame, f"REC {record_label}: [{bar}] {pct}%",
                        (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if record_count >= RECORD_FRAMES:
                recording = False
                print(f"  [DONE] {record_label}: {len(all_windows[record_label])} windows total")

        # UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "FALL DATA COLLECTOR", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)

        fall_count = len(all_windows["Fall"])
        normal_count = len(all_windows["Normal"])
        cv2.putText(frame, f"Fall: {fall_count} | Normal: {normal_count}",
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        status = "RECORDING" if recording else "'f'=fall  'n'=normal  'q'=save"
        cv2.putText(frame, status, (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        cv2.imshow("Fall Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f') and not recording:
            recording = True
            record_label = "Fall"
            record_count = 0
            print("  [REC] Recording FALL — do it now!")
        elif key == ord('n') and not recording:
            recording = True
            record_label = "Normal"
            record_count = 0
            print("  [REC] Recording NORMAL — do your action now!")

    # Save
    total = sum(len(v) for v in all_windows.values())
    if total > 0:
        save_path = os.path.join(DATA_DIR, "fall_windows.json")
        with open(save_path, 'w') as f:
            json.dump(all_windows, f)
        print(f"\n[SAVED] {total} windows → {save_path}")
        print(f"  Fall:   {len(all_windows['Fall'])} windows")
        print(f"  Normal: {len(all_windows['Normal'])} windows")
        print(f"\nNext: python train_fall_model.py")
    else:
        print("[WARN] No data collected!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
