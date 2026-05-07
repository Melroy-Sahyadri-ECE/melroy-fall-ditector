"""
Webcam Pose Data Collector
===========================
Captures real pose data from your webcam for training the activity classifier.

How to use:
  1. Run this script
  2. Press the number key for each activity to START recording:
       1 = Standing
       2 = Sitting  
       3 = Walking
       4 = Lying Down
       5 = Falling (do it quickly!)
       6 = Bending
  3. Hold the pose — it auto-captures 50 frames (~2 seconds)
  4. Do each activity 3-4 times for variety
  5. Press 'q' to quit and save

The script extracts joint angles and body measurements per frame
and saves them for training.
"""

import cv2
import numpy as np
import time
import os
import math
import json
from collections import deque

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_heavy.task")
DATA_DIR = os.path.join(BASE_DIR, "training_data")

# Landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

CLASSES = {
    "1": "Standing",
    "2": "Sitting",
    "3": "Walking",
    "4": "Lying",
    "5": "Falling",
    "6": "Bending",
}

FRAMES_PER_CAPTURE = 50
CAPTURE_INTERVAL = 0.3   # seconds between captures (slow mode)


def angle_3pts(ax, ay, bx, by, cx, cy):
    """Angle at B formed by A-B-C, in degrees."""
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax * bcx + bay * bcy
    mag = math.sqrt(bax**2 + bay**2) * math.sqrt(bcx**2 + bcy**2)
    if mag < 1e-8:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / mag))
    return math.degrees(math.acos(cos_a))


def torso_angle_vertical(sh_cx, sh_cy, hp_cx, hp_cy):
    """Torso angle from vertical (0=upright, 90=horizontal)."""
    dx = sh_cx - hp_cx
    dy = sh_cy - hp_cy
    if abs(dy) < 1e-6:
        return 90.0
    return abs(math.degrees(math.atan2(dx, dy)))


def extract_features(landmarks, frame_h, frame_w, prev_features=None):
    """
    Extract features from landmarks for one frame.
    
    Returns dict with all measurements, or None if insufficient landmarks.
    Features (14 total):
      - torso_angle: torso angle from vertical
      - l_knee_angle, r_knee_angle: knee bend angles
      - l_hip_angle, r_hip_angle: hip angles
      - l_shoulder_angle, r_shoulder_angle: shoulder angles
      - aspect_ratio: body bbox width / height
      - nose_to_hip_ratio: vertical distance nose-to-hip / torso length
      - sh_above_hp: 1 if shoulders above hips, 0 otherwise
      - hip_y_norm: normalized hip Y position
      - torso_velocity: change in torso angle from prev frame
      - hip_velocity: change in hip Y from prev frame
      - body_y_spread: normalized vertical spread of body
    """
    if not landmarks or len(landmarks) < 33:
        return None

    vis_count = sum(1 for lm in landmarks if lm.visibility > 0.5)
    if vis_count < 10:
        return None

    def pt(idx):
        lm = landmarks[idx]
        return lm.x * frame_w, lm.y * frame_h

    nose_x, nose_y = pt(NOSE)
    l_sh_x, l_sh_y = pt(L_SHOULDER)
    r_sh_x, r_sh_y = pt(R_SHOULDER)
    l_el_x, l_el_y = pt(L_ELBOW)
    r_el_x, r_el_y = pt(R_ELBOW)
    l_wr_x, l_wr_y = pt(L_WRIST)
    r_wr_x, r_wr_y = pt(R_WRIST)
    l_hp_x, l_hp_y = pt(L_HIP)
    r_hp_x, r_hp_y = pt(R_HIP)
    l_kn_x, l_kn_y = pt(L_KNEE)
    r_kn_x, r_kn_y = pt(R_KNEE)
    l_an_x, l_an_y = pt(L_ANKLE)
    r_an_x, r_an_y = pt(R_ANKLE)

    sh_cx = (l_sh_x + r_sh_x) / 2
    sh_cy = (l_sh_y + r_sh_y) / 2
    hp_cx = (l_hp_x + r_hp_x) / 2
    hp_cy = (l_hp_y + r_hp_y) / 2

    torso_len = math.sqrt((sh_cx - hp_cx)**2 + (sh_cy - hp_cy)**2)
    if torso_len < 5:
        return None

    # -- Angles --
    t_angle = torso_angle_vertical(sh_cx, sh_cy, hp_cx, hp_cy)
    l_knee = angle_3pts(l_hp_x, l_hp_y, l_kn_x, l_kn_y, l_an_x, l_an_y)
    r_knee = angle_3pts(r_hp_x, r_hp_y, r_kn_x, r_kn_y, r_an_x, r_an_y)
    l_hip = angle_3pts(l_sh_x, l_sh_y, l_hp_x, l_hp_y, l_kn_x, l_kn_y)
    r_hip = angle_3pts(r_sh_x, r_sh_y, r_hp_x, r_hp_y, r_kn_x, r_kn_y)
    l_shoulder = angle_3pts(l_el_x, l_el_y, l_sh_x, l_sh_y, l_hp_x, l_hp_y)
    r_shoulder = angle_3pts(r_el_x, r_el_y, r_sh_x, r_sh_y, r_hp_x, r_hp_y)

    # -- Body ratios --
    core_xs = [l_sh_x, r_sh_x, l_hp_x, r_hp_x, l_kn_x, r_kn_x, l_an_x, r_an_x]
    core_ys = [l_sh_y, r_sh_y, l_hp_y, r_hp_y, l_kn_y, r_kn_y, l_an_y, r_an_y]
    bbox_w = max(core_xs) - min(core_xs)
    bbox_h = max(core_ys) - min(core_ys)
    aspect_ratio = bbox_w / max(bbox_h, 1)
    y_spread = bbox_h / max(frame_h, 1)

    # Nose to hip ratio
    nose_hip_dist = abs(nose_y - hp_cy) / max(torso_len, 1)

    # Shoulders above hips?
    sh_above = 1.0 if sh_cy < hp_cy else 0.0

    # Normalized hip Y
    hip_y_norm = hp_cy / max(frame_h, 1)

    # -- Velocities (from previous frame) --
    torso_vel = 0.0
    hip_vel = 0.0
    if prev_features is not None:
        torso_vel = t_angle - prev_features["torso_angle"]
        hip_vel = hip_y_norm - prev_features["hip_y_norm"]

    features = {
        "torso_angle": t_angle,
        "l_knee_angle": l_knee,
        "r_knee_angle": r_knee,
        "l_hip_angle": l_hip,
        "r_hip_angle": r_hip,
        "l_shoulder_angle": l_shoulder,
        "r_shoulder_angle": r_shoulder,
        "aspect_ratio": aspect_ratio,
        "nose_to_hip_ratio": nose_hip_dist,
        "sh_above_hp": sh_above,
        "hip_y_norm": hip_y_norm,
        "torso_velocity": torso_vel,
        "hip_velocity": hip_vel,
        "body_y_spread": y_spread,
    }
    return features


FEATURE_NAMES = [
    "torso_angle", "l_knee_angle", "r_knee_angle",
    "l_hip_angle", "r_hip_angle",
    "l_shoulder_angle", "r_shoulder_angle",
    "aspect_ratio", "nose_to_hip_ratio", "sh_above_hp",
    "hip_y_norm", "torso_velocity", "hip_velocity", "body_y_spread",
]


def main():
    print("=" * 58)
    print("  POSE DATA COLLECTOR")
    print("=" * 58)
    print()
    print("  Press a number to START recording that activity:")
    print("    1 = Standing    4 = Lying Down")
    print("    2 = Sitting     5 = Falling (do it fast!)")
    print("    3 = Walking     6 = Bending")
    print()
    print("  Each capture records 50 frames (1 every 0.3s = ~15 sec)")
    print("  Do each activity 3-4 times for best results")
    print("  Press 'q' to quit and save")
    print("  Press 'r' to reset all data")
    print()

    if not os.path.exists(POSE_MODEL_PATH):
        print(f"[ERROR] Pose model not found: {POSE_MODEL_PATH}")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    # Async pose
    latest = {"landmarks": None}

    def on_result(result, output_image, timestamp_ms):
        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            latest["landmarks"] = result.pose_landmarks[0]
        else:
            latest["landmarks"] = None

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        result_callback=on_result,
    )

    landmarker = PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Data storage
    all_data = {name: [] for name in CLASSES.values()}
    recording = False
    record_class = ""
    record_count = 0
    prev_features = None
    frame_ts = 0
    last_capture_time = 0

    print("[INFO] Camera opened. Ready to record.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        try:
            landmarker.detect_async(mp_img, frame_ts)
        except Exception:
            pass

        landmarks = latest["landmarks"]

        # Draw skeleton
        if landmarks:
            for i, lm in enumerate(landmarks):
                if lm.visibility > 0.4:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        # Recording (slow capture — 1 frame every CAPTURE_INTERVAL seconds)
        cur_time = time.time()
        if recording and landmarks:
            if cur_time - last_capture_time >= CAPTURE_INTERVAL:
                features = extract_features(landmarks, h, w, prev_features)
                if features is not None:
                    all_data[record_class].append(features)
                    prev_features = features
                    record_count += 1
                    last_capture_time = cur_time

                    if record_count >= FRAMES_PER_CAPTURE:
                        recording = False
                        print(f"  [DONE] Captured {FRAMES_PER_CAPTURE} frames of '{record_class}'"
                              f" (total: {len(all_data[record_class])})")

            # Progress bar (always show while recording)
            if recording:
                pct = record_count * 100 // FRAMES_PER_CAPTURE
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                cv2.putText(frame, f"Recording {record_class}: [{bar}] {record_count}/{FRAMES_PER_CAPTURE}",
                            (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "POSE DATA COLLECTOR", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)

        # Show counts
        y_pos = 50
        for key, name in CLASSES.items():
            count = len(all_data[name])
            color = (0, 255, 0) if count >= FRAMES_PER_CAPTURE else (150, 150, 150)
            cv2.putText(frame, f"{key}={name}: {count}", (15 + (int(key) - 1) * 170, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        if recording:
            cv2.putText(frame, f"RECORDING: {record_class}", (w // 2 - 120, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "'q' save & quit | 'r' reset", (w - 250, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        cv2.imshow("Pose Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            all_data = {name: [] for name in CLASSES.values()}
            print("[INFO] Data reset")
        elif chr(key) in CLASSES and not recording:
            record_class = CLASSES[chr(key)]
            record_count = 0
            recording = True
            prev_features = None
            last_capture_time = 0
            print(f"  [REC] Recording '{record_class}'... hold the pose!")

    # Save data
    total = sum(len(v) for v in all_data.values())
    if total > 0:
        save_path = os.path.join(DATA_DIR, "pose_data.json")
        # Convert to serializable format
        save_data = {}
        for cls, samples in all_data.items():
            save_data[cls] = samples  # list of dicts with float values

        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"\n[SAVED] {total} samples → {save_path}")
        for cls, samples in all_data.items():
            print(f"  {cls:12s}: {len(samples)} frames")
        print(f"\nNext: python train_model.py")
    else:
        print("[WARN] No data captured!")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
