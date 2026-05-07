"""
Room Fall Detector — Raspberry Pi 5 + Hailo AI HAT (13 TOPS)
==============================================================
Uses the hailo-apps (formerly hailo-apps-infra) GStreamer pipeline
for accelerated YOLOv8-pose inference on Hailo-8L.

Requirements (on Pi):
  - Hailo AI HAT+ installed and configured
  - hailo-apps repo cloned and installed:
      git clone https://github.com/hailo-ai/hailo-apps.git
      cd hailo-apps && sudo ./install.sh
  - source setup_env.sh before running

Usage:
  python room_fall_detector_pi.py --input rpi
  python room_fall_detector_pi.py --input /dev/video0
  python room_fall_detector_pi.py --input usb
"""

import os
os.environ["GST_PLUGIN_FEATURE_RANK"] = "vaapidecodebin:NONE"
os.environ["QT_QPA_PLATFORM"] = "xcb" # Fix OpenCV UI on Wayland

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import numpy as np
import cv2
import time
import math
import threading
import urllib.request
from collections import deque

import hailo

from hailo_apps.python.pipeline_apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
)
from hailo_apps.python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.python.core.gstreamer.gstreamer_app import app_callback_class

# ─── COCO 17-keypoint indices ───
NOSE = 0; L_EYE = 1; R_EYE = 2; L_EAR = 3; R_EAR = 4
L_SHOULDER = 5; R_SHOULDER = 6; L_ELBOW = 7; R_ELBOW = 8
L_WRIST = 9; R_WRIST = 10; L_HIP = 11; R_HIP = 12
L_KNEE = 13; R_KNEE = 14; L_ANKLE = 15; R_ANKLE = 16

SKELETON_EDGES = [
    (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP), (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]

MAJOR_BONES = {
    (L_SHOULDER, R_SHOULDER), (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP), (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
}

ACTIVITY_COLORS = {
    "Standing": (0, 230, 118), "Sitting": (255, 200, 0),
    "Walking": (0, 200, 255), "Bending": (0, 255, 200),
    "Lying": (255, 150, 50), "Sleeping": (200, 100, 255),
    "FALL DETECTED": (0, 0, 255), "Unknown": (150, 150, 150),
}


# ─── Geometry helpers ───
def _angle_3pts(ax, ay, bx, by, cx, cy):
    bax, bay = ax - bx, ay - by
    bcx, bcy = cx - bx, cy - by
    dot = bax * bcx + bay * bcy
    mag = math.sqrt(bax**2 + bay**2) * math.sqrt(bcx**2 + bcy**2)
    if mag < 1e-8:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


def _torso_angle(sh_cx, sh_cy, hp_cx, hp_cy):
    dx = sh_cx - hp_cx
    dy = sh_cy - hp_cy
    if abs(dy) < 1e-6:
        return 90.0
    return abs(math.degrees(math.atan2(dx, dy)))


import subprocess

# ─── ntfy.sh Push Notification ───
NTFY_TOPIC = "melroy-fall-detector"  # Change this to your own private topic name
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"
NTFY_COOLDOWN = 30  # Don't send more than 1 alert per 30 seconds
_last_ntfy_time = 0

def send_fall_alert():
    """Send a push notification to the Android app via ntfy.sh.
    Runs in a background thread so it never blocks the video pipeline."""
    global _last_ntfy_time
    now = time.time()
    if now - _last_ntfy_time < NTFY_COOLDOWN:
        return  # Skip — already sent recently
    _last_ntfy_time = now

    def _send():
        try:
            req = urllib.request.Request(
                NTFY_URL,
                data=b"FALL DETECTED! A person has fallen in the monitored room. Please check immediately.",
                headers={
                    "Title": "FALL ALERT - Room Monitor",
                    "Priority": "urgent",
                    "Tags": "warning,rotating_light",
                },
            )
            urllib.request.urlopen(req, timeout=5)
            print(f"[NTFY] Alert sent to topic: {NTFY_TOPIC}")
        except Exception as e:
            print(f"[NTFY] Failed to send alert: {e}")

    threading.Thread(target=_send, daemon=True).start()


# ─── Speaker Alarm ───
ALARM_WAV = "/home/tce/tce/melroy-fall-ditector/alarm.wav"
ALARM_COOLDOWN = 30  # Don't play alarm more than once per 30 seconds
_last_alarm_time = 0
_alarm_process = None

def play_alarm():
    """Play a loud alarm through the connected speaker.
    Runs aplay in the background so it doesn't block the video pipeline."""
    global _last_alarm_time, _alarm_process
    now = time.time()
    if now - _last_alarm_time < ALARM_COOLDOWN:
        return
    # Don't start another alarm if one is still playing
    if _alarm_process is not None and _alarm_process.poll() is None:
        return
    _last_alarm_time = now
    try:
        _alarm_process = subprocess.Popen(
            ["aplay", ALARM_WAV],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("[ALARM] Playing alarm sound!")
    except Exception as e:
        print(f"[ALARM] Failed to play: {e}")


# ─── Per-Person Tracker ───
class PersonTracker:
    HISTORY = 45
    FALL_CONFIRM = 4
    FALL_COOLDOWN = 4.0

    def __init__(self, person_id):
        self.id = person_id
        self.torso_hist = deque(maxlen=self.HISTORY)
        self.hip_y_hist = deque(maxlen=self.HISTORY)
        self.pose_hist = deque(maxlen=self.HISTORY)
        self.time_hist = deque(maxlen=self.HISTORY)
        self.height_hist = deque(maxlen=self.HISTORY)
        self.fall_frames = 0
        self.safe_frames = 0
        self.is_fallen = False
        self.last_fall_time = 0
        self.activity = "Unknown"
        self.last_seen = time.time()
        self.prev_torso = None
        self.prev_hip_y = None
        self.floor_y = None
        self.floor_samples = 0

    def classify(self, keypoints, confs, person_box, frame_h, frame_w):
        self.last_seen = time.time()
        now = self.last_seen
        visible = confs > 0.3
        if visible.sum() < 6:
            return self.activity, self.is_fallen

        def kp(idx):
            return keypoints[idx][0], keypoints[idx][1], confs[idx]

        nose_x, nose_y, nose_c = kp(NOSE)
        l_sh_x, l_sh_y, _ = kp(L_SHOULDER)
        r_sh_x, r_sh_y, _ = kp(R_SHOULDER)
        l_hp_x, l_hp_y, _ = kp(L_HIP)
        r_hp_x, r_hp_y, _ = kp(R_HIP)
        l_kn_x, l_kn_y, _ = kp(L_KNEE)
        r_kn_x, r_kn_y, _ = kp(R_KNEE)
        l_an_x, l_an_y, _ = kp(L_ANKLE)
        r_an_x, r_an_y, _ = kp(R_ANKLE)

        sh_cx = (l_sh_x + r_sh_x) / 2
        sh_cy = (l_sh_y + r_sh_y) / 2
        hp_cx = (l_hp_x + r_hp_x) / 2
        hp_cy = (l_hp_y + r_hp_y) / 2
        torso_len = math.sqrt((sh_cx - hp_cx)**2 + (sh_cy - hp_cy)**2)
        if torso_len < 5:
            return self.activity, self.is_fallen

        t_angle = _torso_angle(sh_cx, sh_cy, hp_cx, hp_cy)
        hip_y_norm = hp_cy / frame_h
        ankle_y_norm = max(l_an_y, r_an_y) / frame_h

        l_knee_a = _angle_3pts(l_hp_x, l_hp_y, l_kn_x, l_kn_y, l_an_x, l_an_y)
        r_knee_a = _angle_3pts(r_hp_x, r_hp_y, r_kn_x, r_kn_y, r_an_x, r_an_y)
        avg_knee = (l_knee_a + r_knee_a) / 2

        vis_xs = keypoints[visible, 0]
        vis_ys = keypoints[visible, 1]
        if len(vis_xs) < 4:
            return self.activity, self.is_fallen
        bbox_w = vis_xs.max() - vis_xs.min()
        bbox_h = vis_ys.max() - vis_ys.min()
        aspect_ratio = bbox_w / max(bbox_h, 1)
        person_height_norm = bbox_h / frame_h

        sh_above_hp = sh_cy < hp_cy
        nose_below_hips = nose_y > hp_cy if nose_c > 0.3 else False

        # Pose classification (original logic)
        pose = "Standing"
        if sh_above_hp:
            if nose_below_hips:
                pose = "Bending"
            elif avg_knee < 115:
                pose = "Sitting"
            else:
                pose = "Standing"
        elif aspect_ratio > 1.0:
            pose = "Lying"
        elif t_angle > 55:
            pose = "Lying"

        # Update floor reference when clearly standing
        if pose == "Standing" and t_angle < 15 and avg_knee > 150:
            if self.floor_y is None:
                self.floor_y = ankle_y_norm
                self.floor_samples = 1
            else:
                self.floor_y = 0.9 * self.floor_y + 0.1 * ankle_y_norm
                self.floor_samples += 1

        # Frame-to-frame velocity (FIX: compute BEFORE updating prev)
        frame_torso_vel = 0.0
        frame_hip_vel = 0.0
        if self.prev_torso is not None and self.prev_hip_y is not None:
            frame_torso_vel = t_angle - self.prev_torso
            frame_hip_vel = hip_y_norm - self.prev_hip_y
        self.prev_torso = t_angle
        self.prev_hip_y = hip_y_norm

        self.torso_hist.append(t_angle)
        self.hip_y_hist.append(hip_y_norm)
        self.height_hist.append(person_height_norm)
        self.pose_hist.append(pose)
        self.time_hist.append(now)

        # Fall scoring (original algorithm)
        fall_score = 0
        n = len(self.torso_hist)

        if frame_torso_vel > 8 and frame_hip_vel > 0.015:
            fall_score += 2
        if frame_torso_vel > 12:
            fall_score += 1
        if frame_hip_vel > 0.02:
            fall_score += 1

        if n >= 4:
            t_list = list(self.torso_hist)
            h_list = list(self.hip_y_hist)
            if t_list[-1] - t_list[-4] > 15 and h_list[-1] - h_list[-4] > 0.04:
                fall_score += 3

        if n >= 8:
            t_list = list(self.torso_hist)
            h_list = list(self.hip_y_hist)
            if t_list[-1] - t_list[-8] > 20 and h_list[-1] - h_list[-8] > 0.06:
                fall_score += 3

        if n >= 5:
            recent = list(self.pose_hist)
            times = list(self.time_hist)
            upright_set = {"Standing", "Walking", "Sitting", "Bending"}
            last_upright_t = 0
            for i in range(len(recent) - 1, -1, -1):
                if recent[i] in upright_set:
                    last_upright_t = times[i]
                    break
            if pose == "Lying" and last_upright_t > 0:
                if 0 < now - last_upright_t < 2.0:
                    fall_score += 3

        # BACKWARD FALL: height collapse
        if n >= 6:
            ht_list = list(self.height_hist)
            max_h = max(ht_list[-6:])
            cur_h = ht_list[-1]
            if max_h > 0.01:
                ratio = cur_h / max_h
                if ratio < 0.5:
                    fall_score += 4
                elif ratio < 0.65:
                    fall_score += 2

        was_upright = False
        if n >= 3:
            recent = list(self.pose_hist)[-15:]
            was_upright = any(p in {"Standing", "Walking", "Sitting", "Bending"} for p in recent)

        # Floor-level gate: ignore falls on beds/couches
        on_floor = True
        if self.floor_y is not None and self.floor_samples >= 5:
            lowest_y = vis_ys.max() / frame_h
            if lowest_y < (self.floor_y - 0.20):
                on_floor = False

        sudden_fall = fall_score >= 4 and was_upright and on_floor

        if sudden_fall:
            self.fall_frames += 3
            self.safe_frames = 0
        elif pose in ("Standing", "Walking", "Sitting", "Bending"):
            self.safe_frames += 1
            self.fall_frames = max(0, self.fall_frames - 2)
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
    def __init__(self):
        self.trackers = {}
        self.scene_had_upright = False
        self.scene_upright_time = 0

    def get_tracker(self, track_id):
        if track_id not in self.trackers:
            tracker = PersonTracker(track_id)
            if self.scene_had_upright and (time.time() - self.scene_upright_time) < 2.0:
                for _ in range(5):
                    tracker.pose_hist.append("Standing")
                    tracker.time_hist.append(self.scene_upright_time)
            self.trackers[track_id] = tracker
        return self.trackers[track_id]

    def update_scene(self):
        for tracker in self.trackers.values():
            if tracker.activity in ("Standing", "Walking", "Sitting", "Bending"):
                self.scene_had_upright = True
                self.scene_upright_time = time.time()
                break

    def cleanup(self, timeout=5.0):
        now = time.time()
        stale = [tid for tid, t in self.trackers.items() if now - t.last_seen > timeout]
        for tid in stale:
            del self.trackers[tid]


# ─── Drawing ───
def draw_skeleton(frame, keypoints, confs, color, min_conf=0.3):
    pts = {}
    for i in range(17):
        if confs[i] > min_conf:
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            pts[i] = (x, y)
    for s, e in SKELETON_EDGES:
        if s in pts and e in pts:
            thickness = 3 if (s, e) in MAJOR_BONES else 2
            cv2.line(frame, pts[s], pts[e], color, thickness, cv2.LINE_AA)
    for idx, (x, y) in pts.items():
        cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)


def draw_person_label(frame, box, activity, track_id, is_fallen):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    color = ACTIVITY_COLORS.get(activity, (150, 150, 150))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    label = f"#{track_id} {activity}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    if is_fallen:
        pulse = int(abs(np.sin(time.time() * 5)) * 200) + 55
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, pulse), 4)


def draw_hud(frame, person_count, any_fall):
    # HUD overlay for person count and fall status
    h, w = frame.shape[:2]
    # Draw a semi-transparent black bar at the top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    # Text info
    cv2.putText(frame, f"People: {person_count}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    status_text = "FALL DETECTED!" if any_fall else "Status: Monitoring"
    status_color = (0, 0, 255) if any_fall else (0, 255, 0)
    cv2.putText(frame, status_text, (w - 300, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)


# ─── Hailo App Callback Class ───
class FallDetectorCallback(app_callback_class):
    def __init__(self):
        super().__init__()
        self.use_frame = True
        self.manager = MultiPersonManager()


# ─── Main Callback (hailo-apps API: element, buffer, user_data) ───
def app_callback(element, buffer, user_data):
    """
    GStreamer probe callback — invoked per-frame by the hailo-apps pipeline.
    Signature: (element, buffer, user_data) — matches hailo-apps >= v26.x
    """
    if buffer is None:
        return

    # Frame count is auto-incremented by the framework wrapper
    frame_count = user_data.get_count()

    # Get video dimensions from the element's source pad
    pad = element.get_static_pad("src")
    format, width, height = get_caps_from_pad(pad)

    frame = None
    if user_data.use_frame and format and width and height:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    person_count = 0
    any_fall = False

    for detection in detections:
        label = detection.get_label()
        if label != "person":
            continue

        confidence = detection.get_confidence()
        if confidence < 0.4:
            continue

        bbox = detection.get_bbox()

        # Get track ID
        track_id = 0
        track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
        if len(track) == 1:
            track_id = track[0].get_id()

        # Get pose landmarks
        landmarks_list = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if len(landmarks_list) == 0:
            continue

        points = landmarks_list[0].get_points()
        if len(points) < 17:
            continue

        # Convert Hailo normalized coords → pixel coords
        keypoints_arr = np.zeros((17, 2), dtype=np.float32)
        confs_arr = np.zeros(17, dtype=np.float32)
        for i in range(17):
            pt = points[i]
            px = (pt.x() * bbox.width() + bbox.xmin()) * width
            py = (pt.y() * bbox.height() + bbox.ymin()) * height
            keypoints_arr[i] = [px, py]
            confs_arr[i] = pt.confidence() if hasattr(pt, 'confidence') else 0.9

        person_box = np.array([
            bbox.xmin() * width, bbox.ymin() * height,
            (bbox.xmin() + bbox.width()) * width,
            (bbox.ymin() + bbox.height()) * height
        ])

        tracker = user_data.manager.get_tracker(track_id)
        activity, is_fallen = tracker.classify(
            keypoints_arr, confs_arr, person_box, height, width
        )
        person_count += 1
        if is_fallen:
            any_fall = True

        if frame is not None:
            color = ACTIVITY_COLORS.get(activity, (0, 255, 0))
            draw_skeleton(frame, keypoints_arr, confs_arr, color)
            draw_person_label(frame, person_box, activity, track_id, is_fallen)

    user_data.manager.update_scene()

    if frame_count % 60 == 0:
        user_data.manager.cleanup()

    if frame is not None:
        draw_hud(frame, person_count, any_fall)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        user_data.set_frame(frame)

    if any_fall:
        print(f"[ALERT] FALL DETECTED! Frame {frame_count}")
        send_fall_alert()
        play_alarm()
    return


if __name__ == "__main__":
    print("=" * 55)
    print("  FALL DETECTOR — Raspberry Pi 5 + Hailo AI HAT")
    print("  Model: yolov8s_pose (13 TOPS, Hailo-8L)")
    print("  API:   hailo-apps (v26.x+)")
    print("=" * 55)

    class CustomPoseApp(GStreamerPoseEstimationApp):
        def get_pipeline_string(self):
            # Use our custom compiled .so with proper threshold
            self.post_process_so = "/home/tce/tce/hailo-rpi5-examples/hailo-apps/hailo_apps/postprocess/build/cpp/libyolov8pose_postprocess.so"
            # Hide the raw GStreamer window — only our custom Fall Detector window shows
            self.video_sink = "fakesink"
            return super().get_pipeline_string()

    user_data = FallDetectorCallback()
    app = CustomPoseApp(app_callback, user_data)
    
    # Enable frame extraction for our custom GUI
    # We keep use_frame = False for the app to avoid the crashing subprocess
    app.options_menu.use_frame = False
    user_data.use_frame = True
    
    from gi.repository import GLib
    import os
    
    # Create a resizable window — OpenCV scales automatically with WINDOW_NORMAL
    cv2.namedWindow("Fall Detector", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Fall Detector", 960, 720)
    
    def gui_update():
        # DRAIN THE QUEUE: Always skip to the latest frame for maximum smoothness
        last_f = None
        while True:
            f = user_data.get_frame()
            if f is None:
                break
            last_f = f
            
        if last_f is not None:
            cv2.imshow("Fall Detector", last_f)
            cv2.waitKey(1)
        return True

    print("Starting pipeline... (Look for 'Fall Detector' window)")
    GLib.timeout_add(33, gui_update)  # ~30fps — matches camera rate
    app.run()
