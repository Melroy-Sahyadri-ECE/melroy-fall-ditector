"""
Body-Invariant Feature Extractor for Activity Recognition
==========================================================
Extracts features from MediaPipe 33-keypoint pose data that are
invariant to body size, distance from camera, and person identity.

Features per frame (26 total):
  - 8 joint angles
  - 8 normalized distances (by torso length)
  - 4 body ratios
  - 3 normalized positions
  - 3 velocities (require previous frame)
"""

import numpy as np
import math

# ── MediaPipe landmark indices ──
NOSE = 0
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24
L_KNEE = 25
R_KNEE = 26
L_ANKLE = 27
R_ANKLE = 28

# Core indices for bounding box (no arms/hands)
CORE_INDICES = [NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
                L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]

NUM_FEATURES = 26  # Features per frame


def _angle_3pts(a, b, c):
    """Angle at point b formed by points a-b-c, in degrees."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cos_angle, -1, 1)))


def _dist(a, b):
    """Euclidean distance between two 2D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _torso_angle_from_vertical(shoulder_mid, hip_mid):
    """Angle of torso from vertical (0 = upright, 90 = horizontal)."""
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    if abs(dy) < 1e-6:
        return 90.0
    return abs(math.degrees(math.atan2(dx, dy)))


def extract_frame_features(keypoints, prev_features=None):
    """
    Extract 26 body-invariant features from one frame of 33 keypoints.

    Args:
        keypoints: array of shape (33, 2) or (33, 3) — (x, y) normalized [0-1]
                   or from CSV (x, y, z). We use only x, y.
        prev_features: features from the previous frame (for velocities).

    Returns:
        numpy array of 26 features, or None if keypoints are insufficient.
    """
    if keypoints is None or len(keypoints) < 33:
        return None

    kp = np.array(keypoints)
    if kp.ndim == 1:
        # Flat array — reshape assuming x, y pairs or x, y, z, vis quads
        if len(kp) == 33 * 4:
            kp = kp.reshape(33, 4)[:, :2]  # Take x, y
        elif len(kp) == 33 * 3:
            kp = kp.reshape(33, 3)[:, :2]
        elif len(kp) == 33 * 2:
            kp = kp.reshape(33, 2)
        else:
            return None

    # Use only x, y
    pts = kp[:, :2].astype(float)

    # Key points
    nose = pts[NOSE]
    l_sh = pts[L_SHOULDER];  r_sh = pts[R_SHOULDER]
    l_el = pts[L_ELBOW];     r_el = pts[R_ELBOW]
    l_wr = pts[L_WRIST];     r_wr = pts[R_WRIST]
    l_hp = pts[L_HIP];       r_hp = pts[R_HIP]
    l_kn = pts[L_KNEE];      r_kn = pts[R_KNEE]
    l_an = pts[L_ANKLE];     r_an = pts[R_ANKLE]

    # Mid-points
    sh_mid = (l_sh + r_sh) / 2
    hp_mid = (l_hp + r_hp) / 2

    # ── Torso length (normalization factor) ──
    torso_len = _dist(sh_mid, hp_mid)
    if torso_len < 1e-6:
        torso_len = 1e-6  # Prevent division by zero

    features = []

    # ── 1. Joint Angles (8 features) ──
    features.append(_angle_3pts(l_el, l_sh, l_hp))    # L shoulder angle
    features.append(_angle_3pts(r_el, r_sh, r_hp))    # R shoulder angle
    features.append(_angle_3pts(l_sh, l_el, l_wr))    # L elbow angle
    features.append(_angle_3pts(r_sh, r_el, r_wr))    # R elbow angle
    features.append(_angle_3pts(l_sh, l_hp, l_kn))    # L hip angle
    features.append(_angle_3pts(r_sh, r_hp, r_kn))    # R hip angle
    features.append(_angle_3pts(l_hp, l_kn, l_an))    # L knee angle
    features.append(_angle_3pts(r_hp, r_kn, r_an))    # R knee angle

    # ── 2. Normalized Distances (8 features) ──
    features.append(_dist(nose, hp_mid) / torso_len)        # Nose to hip center
    features.append(_dist(l_wr, l_hp) / torso_len)          # L hand to L hip
    features.append(_dist(r_wr, r_hp) / torso_len)          # R hand to R hip
    features.append(_dist(l_an, l_hp) / torso_len)          # L foot to L hip
    features.append(_dist(r_an, r_hp) / torso_len)          # R foot to R hip
    features.append(_dist(l_hp, r_hp) / torso_len)          # Hip width
    features.append(_dist(l_sh, r_sh) / torso_len)          # Shoulder width
    # Body height: top to bottom of core body
    core_pts = pts[CORE_INDICES]
    body_h = np.max(core_pts[:, 1]) - np.min(core_pts[:, 1])
    features.append(body_h / torso_len)

    # ── 3. Body Ratios (4 features) ──
    # Torso angle from vertical
    features.append(_torso_angle_from_vertical(sh_mid, hp_mid))
    # Core bbox aspect ratio (width / height)
    bbox_w = np.max(core_pts[:, 0]) - np.min(core_pts[:, 0])
    bbox_h = body_h if body_h > 1e-6 else 1e-6
    features.append(bbox_w / bbox_h)
    # Hip height relative to body (0=top, 1=bottom)
    body_top = np.min(core_pts[:, 1])
    body_bot = np.max(core_pts[:, 1])
    body_range = body_bot - body_top if (body_bot - body_top) > 1e-6 else 1e-6
    features.append((hp_mid[1] - body_top) / body_range)
    # Leg spread ratio (ankle distance / hip distance)
    hip_w = _dist(l_hp, r_hp) if _dist(l_hp, r_hp) > 1e-6 else 1e-6
    features.append(_dist(l_an, r_an) / hip_w)

    # ── 4. Normalized Positions (3 features) ──
    features.append(nose[1])        # Nose Y (normalized 0-1)
    features.append(hp_mid[1])      # Hip center Y
    # Center of mass Y (weighted)
    com_y = sh_mid[1] * 0.3 + hp_mid[1] * 0.5 + (l_kn[1] + r_kn[1]) / 2 * 0.2
    features.append(com_y)

    # ── 5. Velocities (3 features) ──
    if prev_features is not None and len(prev_features) >= 26:
        # Vertical velocity of CoM
        features.append(com_y - prev_features[24])   # CoM Y change
        # Horizontal velocity of CoM
        com_x = sh_mid[0] * 0.3 + hp_mid[0] * 0.5 + (l_kn[0] + r_kn[0]) / 2 * 0.2
        prev_com_x = prev_features[23] * 0.3 + prev_features[23] * 0.5  # approximate
        features.append(com_x - prev_com_x)
        # Torso angular velocity
        features.append(features[16] - prev_features[16])  # torso angle change
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def extract_window_features(frame_features_list):
    """
    Extract statistical features from a window of per-frame features.

    Args:
        frame_features_list: list of numpy arrays, each of shape (26,)

    Returns:
        numpy array of 130 features:
          - last frame features (26)
          - mean over window (26)
          - std over window (26)
          - max-min range (26)
          - delta first-to-last (26)
    """
    if not frame_features_list or len(frame_features_list) < 2:
        return np.zeros(NUM_FEATURES * 5, dtype=np.float32)

    arr = np.array(frame_features_list)  # (window_size, 26)

    current = arr[-1]
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    feat_range = np.max(arr, axis=0) - np.min(arr, axis=0)
    delta = arr[-1] - arr[0]

    return np.concatenate([current, mean, std, feat_range, delta]).astype(np.float32)


WINDOW_FEATURE_SIZE = NUM_FEATURES * 5  # 130
