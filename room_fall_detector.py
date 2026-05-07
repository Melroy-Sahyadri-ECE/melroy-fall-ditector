"""
Room Fall Detector — YOLOv8 + Multi-Person Skeleton
=====================================================
Architecture:
  1. YOLOv8n-pose  →  Detect people + 17 keypoints per person
  2. YOLOv8n       →  Detect beds (COCO class 59)
  3. Per-person temporal tracker  →  Activity classification
  4. Bed-aware logic  →  Sleeping vs Falling

Controls:  q = quit,  s = toggle skeleton/overlay mode
"""

import cv2
import numpy as np
import time
import math
import os
import threading
from collections import deque
from ultralytics import YOLO

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FALL_MODEL_PATH = os.path.join(BASE_DIR, "fall_classifier.joblib")
FALL_SCALER_PATH = os.path.join(BASE_DIR, "fall_scaler.joblib")
WINDOW_SIZE = 15

# ─── Keypoint Indices (COCO 17-keypoint format) ───
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Skeleton connections for drawing
SKELETON_EDGES = [
    # Head
    (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
    # Torso
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP), (L_HIP, R_HIP),
    # Left arm
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    # Right arm
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    # Left leg
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    # Right leg
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]

# Major bones (drawn thicker)
MAJOR_BONES = {
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
}

# Activity colors (BGR)
ACTIVITY_COLORS = {
    "Standing":      (0, 230, 118),    # Green
    "Sitting":       (255, 200, 0),    # Cyan-yellow
    "Walking":       (0, 200, 255),    # Orange
    "Bending":       (0, 255, 200),    # Teal
    "Lying":         (255, 150, 50),   # Light blue
    "Sleeping":      (200, 100, 255),  # Purple
    "FALL DETECTED": (0, 0, 255),      # Red
    "Unknown":       (150, 150, 150),  # Gray
}


# ─── Geometry helpers ───
def _angle_3pts(ax, ay, bx, by, cx, cy):
    """Angle at point B formed by A-B-C, in degrees."""
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax * bcx + bay * bcy
    mag = math.sqrt(bax**2 + bay**2) * math.sqrt(bcx**2 + bcy**2)
    if mag < 1e-8:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _torso_angle(sh_cx, sh_cy, hp_cx, hp_cy):
    """Torso angle from vertical. 0° = upright, 90° = horizontal."""
    dx = sh_cx - hp_cx
    dy = sh_cy - hp_cy
    if abs(dy) < 1e-6:
        return 90.0
    return abs(math.degrees(math.atan2(dx, dy)))


def _bbox_iou(box1, box2):
    """Calculate IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def _bbox_overlap_ratio(person_box, bed_box):
    """How much of the person box overlaps with bed box (0-1)."""
    x1 = max(person_box[0], bed_box[0])
    y1 = max(person_box[1], bed_box[1])
    x2 = min(person_box[2], bed_box[2])
    y2 = min(person_box[3], bed_box[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    person_area = (person_box[2] - person_box[0]) * (person_box[3] - person_box[1])
    return inter / max(person_area, 1e-6)


# ─── ML Fall Model (loaded once, shared) ───
_fall_model = None
_fall_scaler = None

def _load_fall_model():
    global _fall_model, _fall_scaler
    if HAS_JOBLIB and os.path.exists(FALL_MODEL_PATH) and os.path.exists(FALL_SCALER_PATH):
        try:
            _fall_model = joblib.load(FALL_MODEL_PATH)
            _fall_scaler = joblib.load(FALL_SCALER_PATH)
            print("[INFO] ML fall model loaded — using supervised detection")
        except Exception as e:
            print(f"[WARN] Failed to load fall model: {e}")

_load_fall_model()


def _extract_frame_features(keypoints, confs, frame_h, frame_w,
                            sh_cx, sh_cy, hp_cx, hp_cy, t_angle, hip_y_norm,
                            avg_knee, aspect_ratio, sh_above_hp, nose_below_hips,
                            torso_len, bbox_h):
    """Build 8 per-frame features for ML window."""
    return [t_angle, hip_y_norm, avg_knee, aspect_ratio,
            1.0 if sh_above_hp else 0.0,
            1.0 if nose_below_hips else 0.0,
            torso_len / max(frame_h, 1),
            bbox_h / max(frame_h, 1)]


def _predict_fall_window(frame_buffer):
    """Run ML model on a window of frame features. Returns fall probability."""
    if _fall_model is None or len(frame_buffer) < WINDOW_SIZE:
        return 0.0
    window = list(frame_buffer)[-WINDOW_SIZE:]
    arr = np.array(window)
    start, end = arr[0], arr[-1]
    delta = end - start
    velocities = np.diff(arr, axis=0)
    max_vel = velocities.max(axis=0)
    stats = [arr[:, 0].mean(), arr[:, 0].std(),
             arr[:, 1].mean(), arr[:, 1].std(),
             arr[:, 2].mean(), arr[:, 3].mean()]
    features = np.concatenate([start, end, delta, max_vel, stats]).reshape(1, -1)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    X = _fall_scaler.transform(features)
    prob = _fall_model.predict_proba(X)[0][1]  # probability of class 1 (Fall)
    return float(prob)


# ─── Per-Person Activity Tracker ───
class PersonTracker:
    """Tracks temporal activity history for one detected person."""

    HISTORY = 45
    FALL_CONFIRM = 5       # Fewer frames needed — fall is fast
    FALL_COOLDOWN = 4.0

    def __init__(self, person_id):
        self.id = person_id
        self.torso_hist = deque(maxlen=self.HISTORY)
        self.hip_y_hist = deque(maxlen=self.HISTORY)
        self.pose_hist = deque(maxlen=self.HISTORY)
        self.time_hist = deque(maxlen=self.HISTORY)
        self.frame_features = deque(maxlen=60)  # for ML window
        self.fall_frames = 0
        self.safe_frames = 0
        self.is_fallen = False
        self.last_fall_time = 0
        self.activity = "Unknown"
        self.last_seen = time.time()
        self.prev_torso = None
        self.prev_hip_y = None

    def classify(self, keypoints, confs, person_box, bed_boxes, frame_h, frame_w):
        """
        Classify activity for this person.
        keypoints: 17x2 array (x, y pixel coords)
        confs: 17 confidence scores
        person_box: [x1, y1, x2, y2]
        bed_boxes: list of [x1, y1, x2, y2] for beds
        """
        self.last_seen = time.time()

        # Check minimum keypoint visibility
        visible = confs > 0.3
        if visible.sum() < 6:
            return self.activity, self.is_fallen

        # Extract key points
        def kp(idx):
            return keypoints[idx][0], keypoints[idx][1], confs[idx]

        nose_x, nose_y, nose_c = kp(NOSE)
        l_sh_x, l_sh_y, l_sh_c = kp(L_SHOULDER)
        r_sh_x, r_sh_y, r_sh_c = kp(R_SHOULDER)
        l_hp_x, l_hp_y, l_hp_c = kp(L_HIP)
        r_hp_x, r_hp_y, r_hp_c = kp(R_HIP)
        l_kn_x, l_kn_y, l_kn_c = kp(L_KNEE)
        r_kn_x, r_kn_y, r_kn_c = kp(R_KNEE)
        l_an_x, l_an_y, l_an_c = kp(L_ANKLE)
        r_an_x, r_an_y, r_an_c = kp(R_ANKLE)

        # Centers
        sh_cx = (l_sh_x + r_sh_x) / 2
        sh_cy = (l_sh_y + r_sh_y) / 2
        hp_cx = (l_hp_x + r_hp_x) / 2
        hp_cy = (l_hp_y + r_hp_y) / 2

        torso_len = math.sqrt((sh_cx - hp_cx)**2 + (sh_cy - hp_cy)**2)
        if torso_len < 5:
            return self.activity, self.is_fallen

        # ── Measurements ──
        t_angle = _torso_angle(sh_cx, sh_cy, hp_cx, hp_cy)
        hip_y_norm = hp_cy / frame_h

        # Knee angles
        l_knee_a = _angle_3pts(l_hp_x, l_hp_y, l_kn_x, l_kn_y, l_an_x, l_an_y)
        r_knee_a = _angle_3pts(r_hp_x, r_hp_y, r_kn_x, r_kn_y, r_an_x, r_an_y)
        avg_knee = (l_knee_a + r_knee_a) / 2

        # Body aspect ratio
        vis_xs = keypoints[visible, 0]
        vis_ys = keypoints[visible, 1]
        if len(vis_xs) < 4:
            return self.activity, self.is_fallen
        bbox_w = vis_xs.max() - vis_xs.min()
        bbox_h = vis_ys.max() - vis_ys.min()
        aspect_ratio = bbox_w / max(bbox_h, 1)

        # Vertical ordering
        sh_above_hp = sh_cy < hp_cy
        nose_below_hips = nose_y > hp_cy if nose_c > 0.3 else False

        # ── Bed proximity ──
        on_bed = False
        for bed_box in bed_boxes:
            overlap = _bbox_overlap_ratio(person_box, bed_box)
            if overlap > 0.3:
                on_bed = True
                break

        # ── History ──
        now = time.time()
        self.torso_hist.append(t_angle)
        self.hip_y_hist.append(hip_y_norm)
        self.time_hist.append(now)

        # ── Frame-to-frame velocity (instantaneous) ──
        frame_torso_vel = 0.0
        frame_hip_vel = 0.0
        if self.prev_torso is not None:
            frame_torso_vel = t_angle - self.prev_torso
            frame_hip_vel = hip_y_norm - self.prev_hip_y
        self.prev_torso = t_angle
        self.prev_hip_y = hip_y_norm

        # ══════════════════════════════════════
        # POSE CLASSIFICATION
        # ══════════════════════════════════════
        pose = "Standing"

        if sh_above_hp:
            if nose_below_hips:
                pose = "Bending"
            elif avg_knee < 115:
                pose = "Sitting"
            else:
                pose = "Standing"
        elif aspect_ratio > 1.2:
            pose = "Sleeping" if on_bed else "Lying"
        elif t_angle > 60:
            pose = "Sleeping" if on_bed else "Lying"
        else:
            pose = "Standing"

        self.pose_hist.append(pose)

        # ── ML frame features (for trained model) ──
        ff = _extract_frame_features(
            keypoints, confs, frame_h, frame_w,
            sh_cx, sh_cy, hp_cx, hp_cy, t_angle, hip_y_norm,
            avg_knee, aspect_ratio, sh_above_hp, nose_below_hips,
            torso_len, bbox_h)
        if ff is not None:
            self.frame_features.append(ff)

        # ══════════════════════════════════════
        # FALL DETECTION — Multi-signal approach
        # ══════════════════════════════════════
        fall_score = 0  # Accumulate evidence
        n = len(self.torso_hist)

        # ── Signal 1: Frame-to-frame velocity spike ──
        # A single frame with huge velocity = something sudden happened
        if frame_torso_vel > 10 and frame_hip_vel > 0.015:
            fall_score += 2
        if frame_torso_vel > 15:
            fall_score += 1
        if frame_hip_vel > 0.025:
            fall_score += 1

        # ── Signal 2: Short window (4 frames) ──
        if n >= 4:
            t_list = list(self.torso_hist)
            h_list = list(self.hip_y_hist)
            td4 = t_list[-1] - t_list[-4]
            hd4 = h_list[-1] - h_list[-4]
            if td4 > 20 and hd4 > 0.05:
                fall_score += 3

        # ── Signal 3: Medium window (8-12 frames) ──
        if n >= 8:
            t_list = list(self.torso_hist)
            h_list = list(self.hip_y_hist)
            td8 = t_list[-1] - t_list[-8]
            hd8 = h_list[-1] - h_list[-8]
            if td8 > 25 and hd8 > 0.07:
                fall_score += 3

        if n >= 12:
            t_list = list(self.torso_hist)
            h_list = list(self.hip_y_hist)
            td12 = t_list[-1] - t_list[-12]
            hd12 = h_list[-1] - h_list[-12]
            if td12 > 25 and hd12 > 0.08:
                fall_score += 2

        # ── Signal 4: Pose transition (was standing, now lying) ──
        if n >= 5:
            recent = list(self.pose_hist)
            times = list(self.time_hist)
            upright_set = {"Standing", "Walking", "Sitting", "Bending"}
            lying_set = {"Lying", "Sleeping"}

            # Find last upright frame
            last_upright_t = 0
            for i in range(len(recent) - 1, -1, -1):
                if recent[i] in upright_set:
                    last_upright_t = times[i]
                    break

            # If currently lying AND was upright within last 1.5 seconds
            if pose in lying_set and last_upright_t > 0:
                transition_time = now - last_upright_t
                if 0 < transition_time < 1.5:
                    fall_score += 3  # Fast transition = likely fall

        # ── Signal 5: ML model prediction (if trained) ──
        ml_fall_prob = _predict_fall_window(self.frame_features)
        if ml_fall_prob > 0.6:
            fall_score += 4
        elif ml_fall_prob > 0.4:
            fall_score += 2

        # ── Suppress if on bed ──
        if on_bed:
            fall_score = 0

        # ── Was person upright recently? (required for fall) ──
        was_upright = False
        if n >= 3:
            recent = list(self.pose_hist)[-15:]
            upright_set = {"Standing", "Walking", "Sitting", "Bending"}
            was_upright = any(p in upright_set for p in recent)

        sudden_fall = fall_score >= 4 and was_upright

        # ── Fall state machine ──
        if sudden_fall:
            self.fall_frames += 3
            self.safe_frames = 0
        elif pose in ("Standing", "Walking", "Sitting", "Bending"):
            self.safe_frames += 1
            self.fall_frames = max(0, self.fall_frames - 2)
        elif on_bed:
            self.safe_frames += 1
            self.fall_frames = max(0, self.fall_frames - 3)
        else:
            self.safe_frames += 1
            self.fall_frames = max(0, self.fall_frames - 1)

        if self.fall_frames >= self.FALL_CONFIRM:
            self.is_fallen = True
            self.last_fall_time = now
            self.activity = "FALL DETECTED"
        elif self.is_fallen:
            if (now - self.last_fall_time) < self.FALL_COOLDOWN:
                self.activity = "FALL DETECTED"
            elif self.safe_frames > 20:
                self.is_fallen = False
                self.fall_frames = 0
                self.activity = pose
            else:
                self.activity = "FALL DETECTED"
        else:
            self.activity = pose

        return self.activity, self.is_fallen


# ─── Multi-Person Manager ───
class MultiPersonManager:
    """Manages trackers for multiple people using YOLO tracking IDs."""

    def __init__(self):
        self.trackers = {}  # track_id → PersonTracker
        # Scene-level: track if ANY person was upright recently
        self.scene_had_upright = False
        self.scene_upright_time = 0

    def get_tracker(self, track_id):
        if track_id not in self.trackers:
            # New tracker: inherit scene-level upright history
            tracker = PersonTracker(track_id)
            # If someone was upright in the scene recently,
            # seed the new tracker's pose history with "Standing"
            # so fall detection works even after ID reassignment
            if self.scene_had_upright and (time.time() - self.scene_upright_time) < 2.0:
                for _ in range(5):
                    tracker.pose_hist.append("Standing")
                    tracker.time_hist.append(self.scene_upright_time)
            self.trackers[track_id] = tracker
        return self.trackers[track_id]

    def update_scene(self):
        """Update scene-level upright tracking."""
        for tracker in self.trackers.values():
            if tracker.activity in ("Standing", "Walking", "Sitting", "Bending"):
                self.scene_had_upright = True
                self.scene_upright_time = time.time()
                break

    def cleanup(self, timeout=5.0):
        """Remove trackers for people not seen recently."""
        now = time.time()
        stale = [tid for tid, t in self.trackers.items()
                 if now - t.last_seen > timeout]
        for tid in stale:
            del self.trackers[tid]


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
        self.ret = False
        self.frame = None
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
            if self.frame is not None:
                return True, self.frame.copy()
            return False, None

    def release(self):
        self.stopped = True
        self.thread.join(timeout=2)
        self.cap.release()


# ─── Drawing Functions ───
def draw_skeleton(frame, keypoints, confs, color, min_conf=0.3):
    """Draw skeleton on frame with given color."""
    pts = {}
    for i in range(17):
        if confs[i] > min_conf:
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            pts[i] = (x, y)

    # Draw bones
    for s, e in SKELETON_EDGES:
        if s in pts and e in pts:
            thickness = 3 if (s, e) in MAJOR_BONES else 2
            cv2.line(frame, pts[s], pts[e], color, thickness, cv2.LINE_AA)

    # Draw joints
    for idx, (x, y) in pts.items():
        cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)


def draw_person_label(frame, box, activity, track_id, is_fallen):
    """Draw activity label and bounding box around person."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    color = ACTIVITY_COLORS.get(activity, (150, 150, 150))

    # Always draw person bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Label background
    label = f"#{track_id} {activity}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    if is_fallen:
        # Red pulsing border around person
        pulse = int(abs(np.sin(time.time() * 5)) * 200) + 55
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, pulse), 4)


def draw_bed(frame, box):
    """Draw bed detection box."""
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BED", (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2, cv2.LINE_AA)


def draw_hud(frame, fps, person_count, any_fall):
    """Draw top HUD bar."""
    h, w = frame.shape[:2]

    # Dark header bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "ROOM FALL DETECTOR — YOLO + Skeleton", (15, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f} | People: {person_count}", (w - 220, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "'q' quit | 's' toggle skeleton", (w - 250, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

    if any_fall:
        pulse = int(abs(np.sin(time.time() * 4)) * 180) + 75
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, pulse), 5)
        msg = "!! FALL DETECTED — EMERGENCY !!"
        sz = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0]
        tx, ty = (w - sz[0]) // 2, h - 35
        cv2.rectangle(frame, (tx - 15, ty - sz[1] - 15),
                      (tx + sz[0] + 15, ty + 15), (0, 0, 0), -1)
        cv2.rectangle(frame, (tx - 15, ty - sz[1] - 15),
                      (tx + sz[0] + 15, ty + 15), (0, 0, pulse), 2)
        cv2.putText(frame, msg, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)


# ─── Main Loop ───
def main():
    print("=" * 58)
    print("  ROOM FALL DETECTOR — YOLO + Multi-Person Skeleton")
    print("=" * 58)
    print()
    print("  Features:")
    print("    • YOLOv8-Pose: Multi-person skeleton (17 keypoints)")
    print("    • YOLOv8n: Bed detection for sleep/fall distinction")
    print("    • Per-person activity tracking with fall alerts")
    print()
    print("  Controls: q = quit, s = toggle skeleton-only mode")
    print()

    # ── Load models ──
    print("[1/3] Loading YOLOv8n-pose (people + skeleton)...")
    pose_model = YOLO("yolov8n-pose.pt")
    print("[2/3] Loading YOLOv8n (bed detection)...")
    detect_model = YOLO("yolov8n.pt")
    print("[3/3] Opening camera...")

    camera = CameraCapture(src=0)
    if not camera.cap.isOpened():
        print("[ERROR] No camera found!")
        return

    manager = MultiPersonManager()
    overlay_mode = True
    prev_time = time.time()
    fps = 0.0
    frame_count = 0

    print("[INFO] Running! Press 'q' to quit.")

    while True:
        ret, frame = camera.read()
        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # ── Run pose model (people + skeletons) ──
        pose_results = pose_model.track(
            frame, persist=True, verbose=False,
            conf=0.4, iou=0.5,
            classes=[0],  # person only
        )

        # ── Run bed detection every 10 frames (save compute) ──
        bed_boxes = []
        if frame_count % 10 == 0:
            bed_results = detect_model(
                frame, verbose=False,
                conf=0.3, classes=[59],  # bed = class 59
            )
            if bed_results and len(bed_results) > 0:
                for box in bed_results[0].boxes:
                    bed_boxes.append(box.xyxy[0].cpu().numpy())

        # Store bed boxes for frames between detections
        if frame_count % 10 == 0:
            main._cached_beds = bed_boxes
        else:
            bed_boxes = getattr(main, '_cached_beds', [])

        # ── Process each detected person ──
        canvas = frame.copy() if overlay_mode else np.zeros_like(frame)
        person_count = 0
        any_fall = False

        if pose_results and len(pose_results) > 0:
            result = pose_results[0]

            if result.keypoints is not None and result.boxes is not None:
                kps_data = result.keypoints.data.cpu().numpy()   # [N, 17, 3]
                boxes_data = result.boxes.xyxy.cpu().numpy()     # [N, 4]

                # Get track IDs (or use index)
                if result.boxes.id is not None:
                    track_ids = result.boxes.id.cpu().numpy().astype(int)
                else:
                    track_ids = list(range(len(boxes_data)))

                for i in range(len(kps_data)):
                    kps = kps_data[i]           # [17, 3] → x, y, conf
                    keypoints = kps[:, :2]      # [17, 2]
                    confs = kps[:, 2]           # [17]
                    person_box = boxes_data[i]  # [4]
                    track_id = int(track_ids[i]) if i < len(track_ids) else i

                    # Get/create tracker for this person
                    tracker = manager.get_tracker(track_id)
                    activity, is_fallen = tracker.classify(
                        keypoints, confs, person_box, bed_boxes, h, w
                    )
                    person_count += 1
                    if is_fallen:
                        any_fall = True

                    # Draw skeleton with activity color
                    color = ACTIVITY_COLORS.get(activity, (0, 255, 0))
                    draw_skeleton(canvas, keypoints, confs, color)

                    # Draw person label
                    draw_person_label(canvas, person_box, activity, track_id, is_fallen)

        # Draw bed boxes
        for bed_box in bed_boxes:
            draw_bed(canvas, bed_box)

        # Update scene-level tracking (for ID reassignment robustness)
        manager.update_scene()

        # FPS
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - prev_time, 0.001)
        prev_time = now
        frame_count += 1

        # HUD
        draw_hud(canvas, fps, person_count, any_fall)

        # Cleanup old trackers
        if frame_count % 60 == 0:
            manager.cleanup()

        cv2.imshow("Room Fall Detector", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            overlay_mode = not overlay_mode

    camera.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
