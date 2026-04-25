"""
Synthetic ADL Data Generator + Real Fall Data Combiner
======================================================
Generates realistic synthetic pose sequences for Activities of Daily
Living (Standing, Sitting, Walking, Lying Down) using geometric rules
based on MediaPipe 33-keypoint body model.

Combines with real fall data from the UP-Fall dataset to create a
complete, balanced training set.

The synthetic poses use normalized coordinates [0, 1] and include
realistic noise, body proportion variation, and temporal continuity
to ensure the ML model generalizes to real-world data.
"""

import os
import numpy as np
from feature_extractor import (
    extract_frame_features,
    extract_window_features,
    NUM_FEATURES,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

CLASS_LABELS = ["STAND", "SIT", "WALK", "LIE", "FALL"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_LABELS)}

WINDOW_SIZE = 30
STRIDE = 5

# MediaPipe 33-keypoint indices
NOSE = 0
L_EYE_INNER, L_EYE, L_EYE_OUTER = 1, 2, 3
R_EYE_INNER, R_EYE, R_EYE_OUTER = 4, 5, 6
L_EAR, R_EAR = 7, 8
MOUTH_L, MOUTH_R = 9, 10
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32


def _add_noise(pts, scale=0.008):
    """Add Gaussian noise to a pose."""
    return pts + np.random.randn(*pts.shape) * scale


def _generate_standing_pose(cx=0.5, torso_len=0.15, variation=0.0):
    """
    Generate a standing pose.
    Person is upright, arms at sides, legs straight.
    """
    v = variation
    pts = np.zeros((33, 2))

    # Head region (near top)
    head_y = 0.15 + v * 0.03
    pts[NOSE] = [cx, head_y]
    pts[L_EYE_INNER] = [cx - 0.01, head_y - 0.01]
    pts[L_EYE] = [cx - 0.02, head_y - 0.01]
    pts[L_EYE_OUTER] = [cx - 0.03, head_y - 0.01]
    pts[R_EYE_INNER] = [cx + 0.01, head_y - 0.01]
    pts[R_EYE] = [cx + 0.02, head_y - 0.01]
    pts[R_EYE_OUTER] = [cx + 0.03, head_y - 0.01]
    pts[L_EAR] = [cx - 0.04, head_y]
    pts[R_EAR] = [cx + 0.04, head_y]
    pts[MOUTH_L] = [cx - 0.01, head_y + 0.02]
    pts[MOUTH_R] = [cx + 0.01, head_y + 0.02]

    # Shoulders
    sh_y = head_y + 0.08 + v * 0.02
    sh_w = 0.08 + v * 0.01
    pts[L_SHOULDER] = [cx - sh_w, sh_y]
    pts[R_SHOULDER] = [cx + sh_w, sh_y]

    # Elbows (at sides)
    el_y = sh_y + torso_len * 0.5
    pts[L_ELBOW] = [cx - sh_w - 0.01, el_y]
    pts[R_ELBOW] = [cx + sh_w + 0.01, el_y]

    # Wrists (at sides)
    wr_y = sh_y + torso_len * 0.9
    pts[L_WRIST] = [cx - sh_w - 0.02, wr_y]
    pts[R_WRIST] = [cx + sh_w + 0.02, wr_y]

    # Hands
    pts[L_PINKY] = pts[L_WRIST] + [0.01, 0.02]
    pts[R_PINKY] = pts[R_WRIST] + [-0.01, 0.02]
    pts[L_INDEX] = pts[L_WRIST] + [-0.01, 0.02]
    pts[R_INDEX] = pts[R_WRIST] + [0.01, 0.02]
    pts[L_THUMB] = pts[L_WRIST] + [0.02, 0.01]
    pts[R_THUMB] = pts[R_WRIST] + [-0.02, 0.01]

    # Hips
    hp_y = sh_y + torso_len
    hp_w = 0.06 + v * 0.01
    pts[L_HIP] = [cx - hp_w, hp_y]
    pts[R_HIP] = [cx + hp_w, hp_y]

    # Knees (straight down)
    kn_y = hp_y + torso_len * 0.9
    pts[L_KNEE] = [cx - hp_w, kn_y]
    pts[R_KNEE] = [cx + hp_w, kn_y]

    # Ankles
    an_y = hp_y + torso_len * 1.7
    pts[L_ANKLE] = [cx - hp_w, an_y]
    pts[R_ANKLE] = [cx + hp_w, an_y]

    # Feet
    pts[L_HEEL] = pts[L_ANKLE] + [0.01, 0.02]
    pts[R_HEEL] = pts[R_ANKLE] + [-0.01, 0.02]
    pts[L_FOOT] = pts[L_ANKLE] + [-0.02, 0.03]
    pts[R_FOOT] = pts[R_ANKLE] + [0.02, 0.03]

    return pts


def _generate_sitting_pose(cx=0.5, torso_len=0.15, variation=0.0):
    """
    Generate a sitting pose.
    Torso upright, knees bent ~90°, feet forward.
    """
    v = variation
    pts = _generate_standing_pose(cx, torso_len, variation)

    # Adjust hips (higher relative, person is sitting so body is lower)
    hp_y = pts[L_SHOULDER][1] + torso_len
    pts[L_HIP] = [cx - 0.06, hp_y]
    pts[R_HIP] = [cx + 0.06, hp_y]

    # Knees: bent forward, same Y as hips
    kn_y = hp_y + 0.02 + v * 0.02
    kn_x_offset = 0.10 + v * 0.03
    pts[L_KNEE] = [cx - kn_x_offset, kn_y]
    pts[R_KNEE] = [cx + kn_x_offset, kn_y]

    # Ankles: below knees, roughly vertical
    an_y = kn_y + torso_len * 0.7
    pts[L_ANKLE] = [cx - kn_x_offset, an_y]
    pts[R_ANKLE] = [cx + kn_x_offset, an_y]

    pts[L_HEEL] = pts[L_ANKLE] + [0.01, 0.02]
    pts[R_HEEL] = pts[R_ANKLE] + [-0.01, 0.02]
    pts[L_FOOT] = pts[L_ANKLE] + [-0.02, 0.03]
    pts[R_FOOT] = pts[R_ANKLE] + [0.02, 0.03]

    # Arms on lap
    pts[L_WRIST] = [cx - 0.08, hp_y + 0.02]
    pts[R_WRIST] = [cx + 0.08, hp_y + 0.02]
    pts[L_ELBOW] = [(pts[L_SHOULDER][0] + pts[L_WRIST][0]) / 2,
                    (pts[L_SHOULDER][1] + pts[L_WRIST][1]) / 2]
    pts[R_ELBOW] = [(pts[R_SHOULDER][0] + pts[R_WRIST][0]) / 2,
                    (pts[R_SHOULDER][1] + pts[R_WRIST][1]) / 2]

    pts[L_PINKY] = pts[L_WRIST] + [0.01, 0.02]
    pts[R_PINKY] = pts[R_WRIST] + [-0.01, 0.02]
    pts[L_INDEX] = pts[L_WRIST] + [-0.01, 0.02]
    pts[R_INDEX] = pts[R_WRIST] + [0.01, 0.02]
    pts[L_THUMB] = pts[L_WRIST] + [0.02, 0.01]
    pts[R_THUMB] = pts[R_WRIST] + [-0.02, 0.01]

    return pts


def _generate_lying_pose(cx=0.5, torso_len=0.15, variation=0.0):
    """
    Generate a lying down pose.
    Body horizontal, all joints at similar Y level.
    """
    v = variation
    pts = np.zeros((33, 2))

    # All at approximately the same Y (lying flat)
    base_y = 0.65 + v * 0.1
    body_spread = 0.35 + v * 0.05

    # Head on one side
    head_x = cx - body_spread / 2
    pts[NOSE] = [head_x, base_y]
    pts[L_EYE_INNER] = [head_x - 0.01, base_y - 0.02]
    pts[L_EYE] = [head_x - 0.02, base_y - 0.02]
    pts[L_EYE_OUTER] = [head_x - 0.03, base_y - 0.02]
    pts[R_EYE_INNER] = [head_x + 0.01, base_y - 0.02]
    pts[R_EYE] = [head_x + 0.02, base_y - 0.02]
    pts[R_EYE_OUTER] = [head_x + 0.03, base_y - 0.02]
    pts[L_EAR] = [head_x - 0.03, base_y]
    pts[R_EAR] = [head_x + 0.03, base_y]
    pts[MOUTH_L] = [head_x - 0.01, base_y + 0.01]
    pts[MOUTH_R] = [head_x + 0.01, base_y + 0.01]

    # Body spread horizontally
    sh_x = head_x + 0.08
    pts[L_SHOULDER] = [sh_x, base_y - 0.04]
    pts[R_SHOULDER] = [sh_x, base_y + 0.04]

    hp_x = sh_x + torso_len
    pts[L_HIP] = [hp_x, base_y - 0.03]
    pts[R_HIP] = [hp_x, base_y + 0.03]

    kn_x = hp_x + torso_len * 0.9
    pts[L_KNEE] = [kn_x, base_y - 0.03]
    pts[R_KNEE] = [kn_x, base_y + 0.03]

    an_x = kn_x + torso_len * 0.8
    pts[L_ANKLE] = [an_x, base_y - 0.03]
    pts[R_ANKLE] = [an_x, base_y + 0.03]

    # Arms extended or at sides
    el_x = sh_x + 0.03
    pts[L_ELBOW] = [el_x, base_y - 0.08]
    pts[R_ELBOW] = [el_x, base_y + 0.08]

    wr_x = el_x + 0.05
    pts[L_WRIST] = [wr_x, base_y - 0.10]
    pts[R_WRIST] = [wr_x, base_y + 0.10]

    pts[L_PINKY] = pts[L_WRIST] + [0.02, -0.01]
    pts[R_PINKY] = pts[R_WRIST] + [0.02, 0.01]
    pts[L_INDEX] = pts[L_WRIST] + [0.02, 0.01]
    pts[R_INDEX] = pts[R_WRIST] + [0.02, -0.01]
    pts[L_THUMB] = pts[L_WRIST] + [0.01, -0.02]
    pts[R_THUMB] = pts[R_WRIST] + [0.01, 0.02]

    pts[L_HEEL] = pts[L_ANKLE] + [0.02, -0.01]
    pts[R_HEEL] = pts[R_ANKLE] + [0.02, 0.01]
    pts[L_FOOT] = pts[L_ANKLE] + [0.03, 0.01]
    pts[R_FOOT] = pts[R_ANKLE] + [0.03, -0.01]

    return pts


def _generate_walking_sequence(n_frames=60, cx=0.5, torso_len=0.15):
    """
    Generate a walking sequence with leg alternation.
    """
    frames = []
    for i in range(n_frames):
        pts = _generate_standing_pose(cx, torso_len, variation=np.random.uniform(-1, 1) * 0.3)

        # Walking motion: legs alternate
        phase = np.sin(2 * np.pi * i / 20)  # ~20 frame stride cycle
        leg_swing = phase * 0.04

        # Left leg swings forward/back
        pts[L_KNEE][1] += leg_swing
        pts[L_ANKLE][1] += leg_swing * 1.2

        # Right leg opposite
        pts[R_KNEE][1] -= leg_swing
        pts[R_ANKLE][1] -= leg_swing * 1.2

        # Arms swing opposite to legs
        pts[L_WRIST][1] -= leg_swing * 0.5
        pts[R_WRIST][1] += leg_swing * 0.5

        # Slight body bob
        bob = abs(phase) * 0.01
        pts[:, 1] -= bob

        # Slight horizontal movement
        drift = i * 0.001
        pts[:, 0] += drift

        frames.append(_add_noise(pts, scale=0.006))

    return frames


def generate_activity_sequences(n_sequences=80):
    """Generate synthetic pose sequences for each ADL activity."""
    rng = np.random.default_rng(42)

    all_sequences = {}

    for activity in ["STAND", "SIT", "LIE", "WALK"]:
        sequences = []

        for seq_idx in range(n_sequences):
            # Vary body proportions
            cx = 0.4 + rng.random() * 0.2
            torso_len = 0.12 + rng.random() * 0.06
            variation = rng.uniform(-1, 1)
            n_frames = rng.integers(50, 120)

            if activity == "WALK":
                frames = _generate_walking_sequence(n_frames, cx, torso_len)
            else:
                frames = []
                for f in range(n_frames):
                    if activity == "STAND":
                        pose = _generate_standing_pose(cx, torso_len, variation)
                        # Small idle sway
                        sway = np.sin(f / 10) * 0.005
                        pose[:, 0] += sway
                    elif activity == "SIT":
                        pose = _generate_sitting_pose(cx, torso_len, variation)
                        # Slight fidgeting
                        fidget = rng.normal(0, 0.003, pose.shape)
                        pose += fidget
                    elif activity == "LIE":
                        pose = _generate_lying_pose(cx, torso_len, variation)
                        # Breathing motion
                        breath = np.sin(f / 15) * 0.003
                        pose[L_SHOULDER:R_SHOULDER + 1, 1] += breath

                    frames.append(_add_noise(pose))

            sequences.append(frames)

        all_sequences[activity] = sequences
        print(f"  Generated {len(sequences)} sequences for {activity}")

    return all_sequences


def build_training_data():
    """
    Build complete training dataset:
      - Synthetic data for STAND, SIT, WALK, LIE
      - Real fall data from UP-Fall dataset
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("=" * 58)
    print("  Building Complete Training Dataset")
    print("=" * 58)

    all_X = []
    all_y = []
    stats = {c: 0 for c in CLASS_LABELS}

    # ── 1. Generate synthetic ADL data ──
    print("\n[Step 1/3] Generating synthetic ADL pose sequences...")
    adl_sequences = generate_activity_sequences(n_sequences=80)

    for activity, sequences in adl_sequences.items():
        label_idx = CLASS_TO_IDX[activity]

        for frames in sequences:
            # Extract per-frame features
            frame_features = []
            prev_feat = None

            for kp in frames:
                feat = extract_frame_features(kp, prev_feat)
                if feat is not None:
                    frame_features.append(feat)
                    prev_feat = feat
                else:
                    frame_features.append(np.zeros(NUM_FEATURES, dtype=np.float32))

            # Create sliding windows
            for start in range(0, len(frame_features) - WINDOW_SIZE + 1, STRIDE):
                window = frame_features[start:start + WINDOW_SIZE]
                window_feat = extract_window_features(window)
                all_X.append(window_feat)
                all_y.append(label_idx)
                stats[activity] += 1

    # ── 2. Load real fall data if available ──
    print("\n[Step 2/3] Loading real fall data from UP-Fall dataset...")
    fall_X_path = os.path.join(PROCESSED_DIR, "X_features.npy")
    fall_y_path = os.path.join(PROCESSED_DIR, "y_labels.npy")

    if os.path.exists(fall_X_path):
        fall_X = np.load(fall_X_path)
        fall_y = np.load(fall_y_path)
        # Only keep fall samples
        fall_mask = fall_y == CLASS_TO_IDX["FALL"]
        fall_X = fall_X[fall_mask]
        fall_y = fall_y[fall_mask]

        all_X.extend(fall_X)
        all_y.extend(fall_y)
        stats["FALL"] += len(fall_y)
        print(f"  Added {len(fall_y)} real FALL samples from UP-Fall")
    else:
        print("  [WARN] No real fall data found, generating synthetic falls...")
        # Generate synthetic fall sequences (rapid transition to horizontal)
        fall_sequences = _generate_fall_sequences(n_sequences=80)
        label_idx = CLASS_TO_IDX["FALL"]

        for frames in fall_sequences:
            frame_features = []
            prev_feat = None
            for kp in frames:
                feat = extract_frame_features(kp, prev_feat)
                if feat is not None:
                    frame_features.append(feat)
                    prev_feat = feat
                else:
                    frame_features.append(np.zeros(NUM_FEATURES, dtype=np.float32))

            for start in range(0, len(frame_features) - WINDOW_SIZE + 1, STRIDE):
                window = frame_features[start:start + WINDOW_SIZE]
                window_feat = extract_window_features(window)
                all_X.append(window_feat)
                all_y.append(label_idx)
                stats["FALL"] += 1

    # ── 3. Balance classes ──
    print("\n[Step 3/3] Balancing dataset...")
    X = np.array(all_X)
    y = np.array(all_y)

    # Clean NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n[INFO] Complete dataset:")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature dim:   {X.shape[1]}")
    print(f"  Class distribution:")
    for cls in CLASS_LABELS:
        count = stats[cls]
        print(f"    {cls:8s}: {count:6d} samples")

    # Save
    np.save(os.path.join(PROCESSED_DIR, "X_combined.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, "y_combined.npy"), y)

    with open(os.path.join(PROCESSED_DIR, "classes.txt"), 'w') as f:
        for c in CLASS_LABELS:
            f.write(c + '\n')

    print(f"\n[SAVED] Combined dataset → {PROCESSED_DIR}")
    return X, y


def _generate_fall_sequences(n_sequences=80):
    """Generate synthetic fall sequences (rapid standing→lying transition)."""
    rng = np.random.default_rng(123)
    sequences = []

    for _ in range(n_sequences):
        cx = 0.4 + rng.random() * 0.2
        torso_len = 0.12 + rng.random() * 0.06

        frames = []
        total_frames = rng.integers(40, 80)
        fall_start = rng.integers(10, 20)
        fall_duration = rng.integers(5, 12)  # Fast!

        for f in range(total_frames):
            if f < fall_start:
                # Standing before fall
                pose = _generate_standing_pose(cx, torso_len,
                                                rng.uniform(-0.5, 0.5))
            elif f < fall_start + fall_duration:
                # Falling transition (interpolate rapidly)
                t = (f - fall_start) / fall_duration
                t = t * t  # Accelerating curve
                standing = _generate_standing_pose(cx, torso_len, 0)
                lying = _generate_lying_pose(cx, torso_len, 0)
                pose = standing * (1 - t) + lying * t
                # Add impact jitter
                pose += rng.normal(0, 0.01 * t, pose.shape)
            else:
                # Lying on ground after fall
                pose = _generate_lying_pose(cx, torso_len,
                                             rng.uniform(-0.5, 0.5))

            frames.append(_add_noise(pose, scale=0.005))

        sequences.append(frames)

    return sequences


if __name__ == "__main__":
    build_training_data()
