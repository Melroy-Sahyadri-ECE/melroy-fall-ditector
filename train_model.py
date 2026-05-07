"""
Train Activity Classifier from Webcam Data
============================================
Trains a Random Forest on real pose data captured by data_collector.py.

Usage:
    1. First run: python data_collector.py (capture poses)
    2. Then run:  python train_model.py (train the model)
    3. Then run:  python smart_fall_detection.py (live detection)
"""

import os
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "training_data")
MODEL_PATH = os.path.join(BASE_DIR, "activity_classifier.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "feature_scaler.joblib")

CLASS_NAMES = ["Standing", "Sitting", "Walking", "Lying", "Falling", "Bending"]

FEATURE_NAMES = [
    "torso_angle", "l_knee_angle", "r_knee_angle",
    "l_hip_angle", "r_hip_angle",
    "l_shoulder_angle", "r_shoulder_angle",
    "aspect_ratio", "nose_to_hip_ratio", "sh_above_hp",
    "hip_y_norm", "torso_velocity", "hip_velocity", "body_y_spread",
]


def load_data():
    """Load data from data_collector.py output."""
    data_path = os.path.join(DATA_DIR, "pose_data.json")
    if not os.path.exists(data_path):
        print(f"[ERROR] No training data found at {data_path}")
        print("  Run: python data_collector.py")
        return None, None, None

    with open(data_path, 'r') as f:
        raw = json.load(f)

    X_list = []
    y_list = []
    class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    for cls_name, samples in raw.items():
        if cls_name not in class_to_idx:
            print(f"  [SKIP] Unknown class: {cls_name}")
            continue
        idx = class_to_idx[cls_name]
        for sample in samples:
            row = [sample.get(fn, 0.0) for fn in FEATURE_NAMES]
            X_list.append(row)
            y_list.append(idx)

    if not X_list:
        print("[ERROR] No valid samples found!")
        return None, None, None

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, class_to_idx


def train():
    print("=" * 58)
    print("  Activity Classifier Trainer (Real Data)")
    print("=" * 58)

    X, y, class_to_idx = load_data()
    if X is None:
        return

    print(f"\n[INFO] Loaded {len(X)} samples, {X.shape[1]} features")
    present_classes = []
    for name in CLASS_NAMES:
        idx = class_to_idx[name]
        count = np.sum(y == idx)
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {name:12s}: {count:4d} samples")
        if count > 0:
            present_classes.append(name)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    n_splits = min(5, min(np.bincount(y)))
    if n_splits >= 2:
        print(f"\n[Step 1/2] Cross-validation ({n_splits}-fold)...")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        print(f"  CV Accuracy: {scores.mean():.2%} ± {scores.std():.2%}")
    else:
        print("\n[SKIP] Too few samples for cross-validation")

    # Train on full data
    print("\n[Step 2/2] Training on full dataset...")
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    print(f"\n  Training accuracy: {np.mean(y_pred == y):.2%}")
    print()
    target_names = [CLASS_NAMES[i] for i in sorted(np.unique(y))]
    print(classification_report(y, y_pred, target_names=target_names))

    # Feature importance
    print("  Top features:")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(min(8, len(FEATURE_NAMES))):
        print(f"    {FEATURE_NAMES[indices[i]]:22s}: {importances[indices[i]]:.3f}")

    # Save
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # Also save class list
    meta = {
        "classes": CLASS_NAMES,
        "features": FEATURE_NAMES,
        "present_classes": present_classes,
    }
    with open(os.path.join(BASE_DIR, "model_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n[SAVED] Model  → {MODEL_PATH}")
    print(f"[SAVED] Scaler → {SCALER_PATH}")
    print(f"\n[DONE] Run: python smart_fall_detection.py")


if __name__ == "__main__":
    train()
