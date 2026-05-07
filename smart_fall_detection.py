"""
Smart Fall Detection — ML + Angle Velocity
============================================
Uses a trained ML model for static pose classification AND
angle velocity for detecting sudden events (fall, sudden bend).

Two-layer detection:
  Layer 1: ML model classifies current pose (Standing/Sitting/Walking/Lying/Bending)
  Layer 2: Angle velocity detects SUDDEN transitions (fall = rapid angle change)

Controls:  q = quit,  s = toggle overlay mode
"""

import cv2
import numpy as np
import time
import os
import threading
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

import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Use lite model for speed (3x faster than heavy)
POSE_LITE_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")
POSE_HEAVY_PATH = os.path.join(BASE_DIR, "pose_landmarker_heavy.task")
POSE_MODEL_PATH = POSE_LITE_PATH if os.path.exists(POSE_LITE_PATH) else POSE_HEAVY_PATH
ML_MODEL_PATH = os.path.join(BASE_DIR, "activity_classifier.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "feature_scaler.joblib")

# Landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

CLASS_NAMES = ["Standing", "Sitting", "Walking", "Lying", "Falling", "Bending"]

ACTIVITY_COLORS = {
    "Standing":      (0, 255, 0),
    "Sitting":       (255, 200, 0),
    "Walking":       (0, 200, 255),
    "Lying":         (255, 150, 50),
    "Bending":       (0, 255, 200),
    "Falling":       (0, 0, 255),
    "FALL DETECTED": (0, 0, 255),
    "No Person":     (100, 100, 100),
    "No Model":      (100, 100, 100),
    "Initializing":  (100, 100, 100),
}

SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

MAJOR_BONES = {
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (24, 26), (26, 28),
}

FEATURE_NAMES = [
    "torso_angle", "l_knee_angle", "r_knee_angle",
    "l_hip_angle", "r_hip_angle",
    "l_shoulder_angle", "r_shoulder_angle",
    "aspect_ratio", "nose_to_hip_ratio", "sh_above_hp",
    "hip_y_norm", "torso_velocity", "hip_velocity", "body_y_spread",
]


# ─── Threaded Camera ───
class CameraCapture:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1 if src == 0 else 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = False, None
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return (self.ret, self.frame.copy()) if self.frame is not None else (False, None)

    def is_opened(self): return self.cap.isOpened()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=2)
        self.cap.release()


# ─── Geometry ───
def _angle_3pts(ax, ay, bx, by, cx, cy):
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax * bcx + bay * bcy
    mag = math.sqrt(bax**2 + bay**2) * math.sqrt(bcx**2 + bcy**2)
    if mag < 1e-8: return 0.0
    return math.degrees(math.acos(max(-1, min(1, dot / mag))))

def _torso_angle(sh_cx, sh_cy, hp_cx, hp_cy):
    dx, dy = sh_cx - hp_cx, sh_cy - hp_cy
    if abs(dy) < 1e-6: return 90.0
    return abs(math.degrees(math.atan2(dx, dy)))


# ─── Feature Extraction (same as data_collector.py) ───
def extract_features(landmarks, frame_h, frame_w, prev_features=None):
    if not landmarks or len(landmarks) < 33:
        return None

    if sum(1 for lm in landmarks if lm.visibility > 0.5) < 10:
        return None

    def pt(idx):
        lm = landmarks[idx]
        return lm.x * frame_w, lm.y * frame_h

    nose_x, nose_y = pt(NOSE)
    l_sh_x, l_sh_y = pt(L_SHOULDER); r_sh_x, r_sh_y = pt(R_SHOULDER)
    l_el_x, l_el_y = pt(L_ELBOW);   r_el_x, r_el_y = pt(R_ELBOW)
    l_wr_x, l_wr_y = pt(L_WRIST);   r_wr_x, r_wr_y = pt(R_WRIST)
    l_hp_x, l_hp_y = pt(L_HIP);     r_hp_x, r_hp_y = pt(R_HIP)
    l_kn_x, l_kn_y = pt(L_KNEE);    r_kn_x, r_kn_y = pt(R_KNEE)
    l_an_x, l_an_y = pt(L_ANKLE);   r_an_x, r_an_y = pt(R_ANKLE)

    sh_cx, sh_cy = (l_sh_x + r_sh_x) / 2, (l_sh_y + r_sh_y) / 2
    hp_cx, hp_cy = (l_hp_x + r_hp_x) / 2, (l_hp_y + r_hp_y) / 2

    torso_len = math.sqrt((sh_cx - hp_cx)**2 + (sh_cy - hp_cy)**2)
    if torso_len < 5: return None

    t_angle = _torso_angle(sh_cx, sh_cy, hp_cx, hp_cy)
    l_knee = _angle_3pts(l_hp_x, l_hp_y, l_kn_x, l_kn_y, l_an_x, l_an_y)
    r_knee = _angle_3pts(r_hp_x, r_hp_y, r_kn_x, r_kn_y, r_an_x, r_an_y)
    l_hip = _angle_3pts(l_sh_x, l_sh_y, l_hp_x, l_hp_y, l_kn_x, l_kn_y)
    r_hip = _angle_3pts(r_sh_x, r_sh_y, r_hp_x, r_hp_y, r_kn_x, r_kn_y)
    l_sh_a = _angle_3pts(l_el_x, l_el_y, l_sh_x, l_sh_y, l_hp_x, l_hp_y)
    r_sh_a = _angle_3pts(r_el_x, r_el_y, r_sh_x, r_sh_y, r_hp_x, r_hp_y)

    core_xs = [l_sh_x, r_sh_x, l_hp_x, r_hp_x, l_kn_x, r_kn_x, l_an_x, r_an_x]
    core_ys = [l_sh_y, r_sh_y, l_hp_y, r_hp_y, l_kn_y, r_kn_y, l_an_y, r_an_y]
    bbox_w = max(core_xs) - min(core_xs)
    bbox_h = max(core_ys) - min(core_ys)
    aspect_ratio = bbox_w / max(bbox_h, 1)
    y_spread = bbox_h / max(frame_h, 1)
    nose_hip_dist = abs(nose_y - hp_cy) / max(torso_len, 1)
    sh_above = 1.0 if sh_cy < hp_cy else 0.0
    hip_y_norm = hp_cy / max(frame_h, 1)

    torso_vel = 0.0
    hip_vel = 0.0
    if prev_features is not None:
        torso_vel = t_angle - prev_features["torso_angle"]
        hip_vel = hip_y_norm - prev_features["hip_y_norm"]

    return {
        "torso_angle": t_angle,
        "l_knee_angle": l_knee, "r_knee_angle": r_knee,
        "l_hip_angle": l_hip, "r_hip_angle": r_hip,
        "l_shoulder_angle": l_sh_a, "r_shoulder_angle": r_sh_a,
        "aspect_ratio": aspect_ratio,
        "nose_to_hip_ratio": nose_hip_dist,
        "sh_above_hp": sh_above,
        "hip_y_norm": hip_y_norm,
        "torso_velocity": torso_vel,
        "hip_velocity": hip_vel,
        "body_y_spread": y_spread,
    }


# ─── Classifier ───
class SmartClassifier:
    """
    Two-layer classification:
      Layer 1: ML model for static pose
      Layer 2: Angle velocity for sudden events
    """

    FALL_CONFIRM = 8
    FALL_COOLDOWN = 3.0
    HISTORY = 30

    def __init__(self):
        self.model = None
        self.scaler = None
        self.has_model = False

        if os.path.exists(ML_MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.model = joblib.load(ML_MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
                self.has_model = True
                print("[INFO] ML classifier loaded")
            except Exception as e:
                print(f"[WARN] ML model failed: {e}")
        else:
            print("[WARN] No ML model. Run data_collector.py then train_model.py")

        self.prev_features = None
        self.torso_hist = deque(maxlen=self.HISTORY)
        self.hip_y_hist = deque(maxlen=self.HISTORY)
        self.pose_hist = deque(maxlen=self.HISTORY)

        self.is_fallen = False
        self.last_fall_time = 0
        self.fall_frames = 0
        self.safe_frames = 0
        self.activity = "Initializing"
        self.confidence = 0.0

    def classify(self, landmarks, frame_h, frame_w):
        features = extract_features(landmarks, frame_h, frame_w, self.prev_features)
        if features is None:
            return self.activity, self.confidence, self.is_fallen, {}

        self.prev_features = features

        # Track history
        self.torso_hist.append(features["torso_angle"])
        self.hip_y_hist.append(features["hip_y_norm"])

        # ═══ LAYER 1: ML Classification ═══
        ml_pose = "Standing"
        ml_conf = 50.0
        ml_probs = {}

        if self.has_model:
            X = np.array([[features[fn] for fn in FEATURE_NAMES]], dtype=np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = self.scaler.transform(X)
            pred = self.model.predict(X_scaled)[0]
            ml_pose = CLASS_NAMES[pred]

            try:
                probs = self.model.predict_proba(X_scaled)[0]
                ml_conf = float(probs[pred]) * 100
                for i, p in enumerate(probs):
                    if p > 0.05:
                        ml_probs[CLASS_NAMES[i]] = round(float(p) * 100)
            except Exception:
                ml_conf = 80.0

        self.pose_hist.append(ml_pose)

        # ═══ LAYER 2: Angle Velocity + ML Falling ═══
        # Two paths to trigger fall:
        #   Path A: Strong velocity (hip drop > 8% AND torso change > 25°)
        #   Path B: ML says "Falling" AND moderate velocity (hip drop > 4%)
        sudden_fall = False
        now = time.time()
        hip_delta = 0.0
        torso_delta = 0.0

        if len(self.torso_hist) >= 8 and len(self.hip_y_hist) >= 8:
            torso_list = list(self.torso_hist)
            hip_list = list(self.hip_y_hist)

            torso_delta = torso_list[-1] - torso_list[-8]
            hip_delta = hip_list[-1] - hip_list[-8]

            # Was person upright recently? (3+ consecutive frames)
            recent = list(self.pose_hist)[-15:]
            upright_set = {"Standing", "Walking", "Sitting", "Bending"}
            consec = 0
            was_upright = False
            for p in recent:
                if p in upright_set:
                    consec += 1
                    if consec >= 3:
                        was_upright = True
                        break
                else:
                    consec = 0

            # Path A: Strong velocity alone
            if was_upright and hip_delta > 0.08 and torso_delta > 25:
                sudden_fall = True

            # Path B: ML says Falling + moderate velocity
            if was_upright and ml_pose == "Falling" and hip_delta > 0.04:
                sudden_fall = True

        # ═══ Fall State Machine ═══
        if sudden_fall:
            self.fall_frames += 2
            self.safe_frames = 0
        elif ml_pose == "Falling" and hip_delta > 0.02:
            # ML says falling with slight movement — weak signal
            self.fall_frames += 1
            self.safe_frames = 0
        elif ml_pose in ("Standing", "Walking", "Sitting", "Bending"):
            self.safe_frames += 1
            self.fall_frames = max(0, self.fall_frames - 2)
        elif ml_pose == "Lying":
            self.safe_frames += 1
            self.fall_frames = max(0, self.fall_frames - 1)
        else:
            # ML says "Falling" but no velocity at all → just lying still
            self.safe_frames += 1
            self.fall_frames = max(0, self.fall_frames - 1)

        if self.fall_frames >= self.FALL_CONFIRM:
            self.is_fallen = True
            self.last_fall_time = now
            activity = "FALL DETECTED"
            conf = 95.0
        elif self.is_fallen:
            if (now - self.last_fall_time) < self.FALL_COOLDOWN:
                activity = "FALL DETECTED"
                conf = 90.0
            elif self.safe_frames > 20:
                self.is_fallen = False
                self.fall_frames = 0
                activity = ml_pose
                conf = ml_conf
            else:
                activity = "FALL DETECTED"
                conf = 85.0
        else:
            # Remap static "Falling" to "Lying" (no motion = just lying)
            if ml_pose == "Falling" and abs(hip_delta) < 0.02:
                activity = "Lying"
                conf = ml_conf
            else:
                activity = ml_pose
                conf = ml_conf

        self.activity = activity
        self.confidence = conf

        debug = {
            "t": f"{features['torso_angle']:.0f}°",
            "tv": f"{features['torso_velocity']:+.1f}",
            "hv": f"{features['hip_velocity']:+.3f}",
            "fall": self.fall_frames,
        }

        return activity, conf, self.is_fallen, debug


# ─── Drawing ───
def draw_skeleton(frame, landmarks, h, w, activity, overlay):
    canvas = frame.copy() if overlay else np.zeros_like(frame)
    if not landmarks: return canvas
    color = ACTIVITY_COLORS.get(activity, (0, 255, 0))
    pts = {}
    for idx, lm in enumerate(landmarks):
        if lm.visibility > 0.4:
            pts[idx] = (int(lm.x * w), int(lm.y * h))
    for s, e in SKELETON_CONNECTIONS:
        if s in pts and e in pts:
            t = 4 if (s, e) in MAJOR_BONES else 3
            cv2.line(canvas, pts[s], pts[e], color, t, cv2.LINE_AA)
    for idx, (x, y) in pts.items():
        cv2.circle(canvas, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)
    return canvas


def draw_hud(frame, activity, conf, is_fallen, debug, fps, mode):
    h, w = frame.shape[:2]
    color = ACTIVITY_COLORS.get(activity, (150, 150, 150))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "SMART FALL DETECTION — ML+Velocity", (15, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f} | {mode}", (w - 200, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

    if is_fallen:
        pulse = int(abs(np.sin(time.time() * 5)) * 200) + 55
        color = (0, 0, pulse)

    cv2.putText(frame, f"Activity: {activity} ({conf:.0f}%)", (15, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    if debug:
        info = " | ".join(f"{k}:{v}" for k, v in debug.items())
        cv2.putText(frame, info, (15, 73),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

    cv2.putText(frame, "'q' quit | 's' toggle", (w - 150, 73),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100, 100, 100), 1, cv2.LINE_AA)

    if is_fallen:
        pulse = int(abs(np.sin(time.time() * 4)) * 180) + 75
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, pulse), 5)
        msg = "!! FALL DETECTED — EMERGENCY !!"
        sz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0]
        tx, ty = (w - sz[0]) // 2, h - 35
        cv2.rectangle(frame, (tx - 15, ty - sz[1] - 15), (tx + sz[0] + 15, ty + 15), (0, 0, 0), -1)
        cv2.rectangle(frame, (tx - 15, ty - sz[1] - 15), (tx + sz[0] + 15, ty + 15), (0, 0, pulse), 2)
        cv2.putText(frame, msg, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
    elif activity == "Lying":
        cv2.putText(frame, "Person resting (no emergency)", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 80), 1, cv2.LINE_AA)

    return frame


# ─── Main ───
def main():
    print("=" * 58)
    print("  SMART FALL DETECTION — ML + Angle Velocity")
    print("=" * 58)
    print("  Controls: q = quit, s = toggle overlay")
    print()

    if not os.path.exists(POSE_MODEL_PATH):
        print(f"[ERROR] Pose model not found: {POSE_MODEL_PATH}")
        return

    latest = {"landmarks": None}
    def on_result(result, output_image, timestamp_ms):
        latest["landmarks"] = result.pose_landmarks[0] if result.pose_landmarks else None

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM, num_poses=1,
        min_pose_detection_confidence=0.6, min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6, result_callback=on_result,
    )

    landmarker = PoseLandmarker.create_from_options(options)
    classifier = SmartClassifier()

    print("[INFO] Opening camera...")
    camera = CameraCapture(src=0)
    if not camera.is_opened():
        print("[ERROR] No camera!")
        landmarker.close()
        return

    overlay_mode = True
    prev_time = time.time()
    fps = 0.0
    frame_ts = 0

    print("[INFO] Running...")

    while True:
        ret, frame = camera.read()
        if not ret or frame is None: continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 33
        try: landmarker.detect_async(mp_img, frame_ts)
        except: pass

        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev_time, 0.001)
        prev_time = now

        landmarks = latest["landmarks"]
        if landmarks:
            activity, conf, is_fallen, debug = classifier.classify(landmarks, h, w)
            vis = draw_skeleton(frame, landmarks, h, w, activity, overlay_mode)
        else:
            vis = frame.copy() if overlay_mode else np.zeros_like(frame)
            activity, conf, is_fallen, debug = "No Person", 0, False, {}

        mode_name = "Overlay" if overlay_mode else "Skeleton"
        output = draw_hud(vis, activity, conf, is_fallen, debug, fps, mode_name)
        cv2.imshow("Smart Fall Detection", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('s'):
            overlay_mode = not overlay_mode

    camera.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
