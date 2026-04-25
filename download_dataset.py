"""
UP-Fall Dataset Downloader & Processor
=======================================
Downloads the 3D Skeletons UP-Fall Dataset from GitHub and processes
the CSV files for training an activity recognition model.

Activities:
  A1-A5: Falls (forward, knees, backward, sideways, sitting-fall)
  A6: Walking
  A7: Standing
  A8: Sitting
  A9: Picking up object → mapped to Standing
  A10: Jumping → mapped to Walking
  A11: Lying down
"""

import os
import io
import zipfile
import requests
import pandas as pd
import numpy as np
from feature_extractor import extract_frame_features, extract_window_features, NUM_FEATURES

# ── Configuration ──
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")

GITHUB_BASE = "https://github.com/Tresor-Koffi/3D_skeletons-UP-Fall-Dataset/raw/main"
SUBJECT_FILES = [f"SUBJECT{i}.zip" for i in range(1, 6)]

WINDOW_SIZE = 30  # frames per window
STRIDE = 5        # slide by 5 frames

# Activity mapping: original activity ID → simplified class
ACTIVITY_MAP = {
    1: "FALL",     # Falling forward using hands
    2: "FALL",     # Falling forward using knees
    3: "FALL",     # Falling backwards
    4: "FALL",     # Falling sideways
    5: "FALL",     # Falling sitting in empty chair
    6: "WALK",     # Walking
    7: "STAND",    # Standing
    8: "SIT",      # Sitting
    9: "STAND",    # Picking up object → standing
    10: "WALK",    # Jumping → walking
    11: "LIE",     # Lying down
}

CLASS_LABELS = ["STAND", "SIT", "WALK", "LIE", "FALL"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_LABELS)}


def download_dataset():
    """Download all subject ZIP files from GitHub."""
    os.makedirs(DATASET_DIR, exist_ok=True)

    for fname in SUBJECT_FILES:
        fpath = os.path.join(DATASET_DIR, fname)
        if os.path.exists(fpath):
            print(f"  [SKIP] {fname} already exists")
            continue

        url = f"{GITHUB_BASE}/{fname}"
        print(f"  [DOWNLOAD] {fname} from GitHub...")
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            downloaded = 0

            with open(fpath, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded * 100 // total
                        print(f"\r    Progress: {pct}% ({downloaded // 1024}KB / {total // 1024}KB)", end="")
            print()
            print(f"  [DONE] {fname}")
        except Exception as e:
            print(f"  [ERROR] Failed to download {fname}: {e}")
            if os.path.exists(fpath):
                os.remove(fpath)
            raise


def extract_zips():
    """Extract all subject ZIP files."""
    csv_dir = os.path.join(DATASET_DIR, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    for fname in SUBJECT_FILES:
        fpath = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [SKIP] {fname} not found")
            continue

        print(f"  [EXTRACT] {fname}...")
        try:
            with zipfile.ZipFile(fpath, 'r') as zf:
                zf.extractall(csv_dir)
            print(f"  [DONE] Extracted {fname}")
        except Exception as e:
            print(f"  [ERROR] Failed to extract {fname}: {e}")


def find_csv_files(base_dir):
    """Recursively find all CSV files in the dataset directory."""
    csv_files = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.csv') and f[0] == 'C':
                csv_files.append(os.path.join(root, f))
    return sorted(csv_files)


def parse_filename(filepath):
    """Extract camera, subject, activity, trial from filename."""
    fname = os.path.basename(filepath).replace('.csv', '')
    try:
        # Format: C{cam}S{subj}A{act}T{trial}
        parts = {}
        remaining = fname
        for prefix in ['C', 'S', 'A', 'T']:
            idx = remaining.index(prefix)
            remaining = remaining[idx + 1:]
            # Find the next letter or end
            num = ''
            for ch in remaining:
                if ch.isdigit():
                    num += ch
                else:
                    break
            parts[prefix] = int(num)
            remaining = remaining[len(num):]
        return parts['C'], parts['S'], parts['A'], parts['T']
    except (ValueError, KeyError):
        return None, None, None, None


def load_csv_keypoints(filepath):
    """
    Load a CSV file and extract the 33 keypoints per frame.
    Returns array of shape (num_frames, 33, 2) — x, y coordinates.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  [ERROR] Cannot read {filepath}: {e}")
        return None

    frames = []
    for _, row in df.iterrows():
        values = row.values
        # Try to find the keypoint columns
        # Format: frame_num, then 33 * (x, y, z, vis), then impact, timestamp
        # So keypoints start at index 1, each has 4 values
        if len(values) >= 1 + 33 * 4:
            kp_data = values[1:1 + 33 * 4]
            kp_array = np.array(kp_data, dtype=float).reshape(33, 4)
            # Take (x, y) only
            frames.append(kp_array[:, :2])
        elif len(values) >= 33 * 4:
            kp_data = values[:33 * 4]
            kp_array = np.array(kp_data, dtype=float).reshape(33, 4)
            frames.append(kp_array[:, :2])
        elif len(values) >= 33 * 3:
            kp_data = values[:33 * 3]
            kp_array = np.array(kp_data, dtype=float).reshape(33, 3)
            frames.append(kp_array[:, :2])
        elif len(values) >= 33 * 2:
            kp_data = values[:33 * 2]
            kp_array = np.array(kp_data, dtype=float).reshape(33, 2)
            frames.append(kp_array[:, :2])

    if len(frames) == 0:
        return None

    return np.array(frames)


def process_dataset():
    """
    Process all CSV files → extract features → create sliding windows → save.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    csv_dir = os.path.join(DATASET_DIR, "csv")
    csv_files = find_csv_files(csv_dir)

    if not csv_files:
        # Try the dataset dir directly
        csv_files = find_csv_files(DATASET_DIR)

    if not csv_files:
        print("[ERROR] No CSV files found! Run download first.")
        return

    print(f"\n[INFO] Found {len(csv_files)} CSV files")

    all_features = []
    all_labels = []
    stats = {c: 0 for c in CLASS_LABELS}

    for csv_path in csv_files:
        cam, subj, act, trial = parse_filename(csv_path)
        if act is None or act not in ACTIVITY_MAP:
            continue

        activity_class = ACTIVITY_MAP[act]
        label_idx = CLASS_TO_IDX[activity_class]

        # Load keypoints
        keypoints = load_csv_keypoints(csv_path)
        if keypoints is None or len(keypoints) < WINDOW_SIZE:
            continue

        # Extract per-frame features
        frame_features = []
        prev_feat = None
        for i in range(len(keypoints)):
            feat = extract_frame_features(keypoints[i], prev_feat)
            if feat is not None:
                frame_features.append(feat)
                prev_feat = feat
            else:
                frame_features.append(np.zeros(NUM_FEATURES, dtype=np.float32))

        # Create sliding windows
        for start in range(0, len(frame_features) - WINDOW_SIZE + 1, STRIDE):
            window = frame_features[start:start + WINDOW_SIZE]
            window_feat = extract_window_features(window)
            all_features.append(window_feat)
            all_labels.append(label_idx)
            stats[activity_class] += 1

    if not all_features:
        print("[ERROR] No features extracted!")
        return

    X = np.array(all_features)
    y = np.array(all_labels)

    print(f"\n[INFO] Dataset processed:")
    print(f"  Total samples: {len(X)}")
    print(f"  Feature dim:   {X.shape[1]}")
    print(f"  Class distribution:")
    for cls, count in stats.items():
        print(f"    {cls:8s}: {count:6d} samples")

    # Save
    np.save(os.path.join(PROCESSED_DIR, "X_features.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, "y_labels.npy"), y)

    # Save class names
    with open(os.path.join(PROCESSED_DIR, "classes.txt"), 'w') as f:
        for c in CLASS_LABELS:
            f.write(c + '\n')

    print(f"\n[SAVED] Features → {PROCESSED_DIR}")
    return X, y


def main():
    print("=" * 58)
    print("  UP-Fall Dataset Downloader & Processor")
    print("=" * 58)

    print("\n[Step 1/3] Downloading dataset from GitHub...")
    download_dataset()

    print("\n[Step 2/3] Extracting ZIP files...")
    extract_zips()

    print("\n[Step 3/3] Processing CSV files → feature extraction...")
    process_dataset()

    print("\n[COMPLETE] Ready to train! Run: python train_model.py")


if __name__ == "__main__":
    main()
