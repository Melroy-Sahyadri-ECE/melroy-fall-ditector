"""
Train Fall Detection Model
============================
Trains a Random Forest classifier on windowed skeleton features
collected by fall_data_collector.py.

Usage:
  1. python fall_data_collector.py  (collect data)
  2. python train_fall_model.py     (train model)
  3. python room_fall_detector.py   (run with ML fall detection)
"""

import os
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "fall_training_data")
MODEL_PATH = os.path.join(BASE_DIR, "fall_classifier.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "fall_scaler.joblib")


def main():
    print("=" * 58)
    print("  Fall Detection Model Trainer")
    print("=" * 58)

    data_path = os.path.join(DATA_DIR, "fall_windows.json")
    if not os.path.exists(data_path):
        print(f"\n[ERROR] No training data at {data_path}")
        print("  Run: python fall_data_collector.py")
        return

    with open(data_path, 'r') as f:
        raw = json.load(f)

    fall_windows = raw.get("Fall", [])
    normal_windows = raw.get("Normal", [])

    print(f"\n[INFO] Data loaded:")
    print(f"  Fall:   {len(fall_windows)} windows")
    print(f"  Normal: {len(normal_windows)} windows")

    if len(fall_windows) < 5 or len(normal_windows) < 5:
        print("\n[ERROR] Need at least 5 windows of each class!")
        print("  Record more data with fall_data_collector.py")
        return

    # Build X, y
    X_fall = np.array(fall_windows, dtype=np.float32)
    X_normal = np.array(normal_windows, dtype=np.float32)

    X = np.vstack([X_fall, X_normal])
    y = np.concatenate([np.ones(len(X_fall)), np.zeros(len(X_normal))])

    # Clean
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"\n[INFO] Total: {len(X)} samples, {X.shape[1]} features")

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation
    n_splits = min(5, min(len(fall_windows), len(normal_windows)))
    if n_splits >= 2:
        print(f"\n[Step 1/2] Cross-validation ({n_splits}-fold)...")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        print(f"  CV Accuracy: {scores.mean():.1%} ± {scores.std():.1%}")

    # Train on full data
    print("\n[Step 2/2] Training on full dataset...")
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    print(f"\n  Training accuracy: {np.mean(y_pred == y):.1%}")
    print()
    print(classification_report(y, y_pred, target_names=["Normal", "Fall"]))

    # Feature importance
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("  Top features by importance:")
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. Feature[{idx}]: {importances[idx]:.3f}")

    # Save
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\n[SAVED] Model  → {MODEL_PATH}")
    print(f"[SAVED] Scaler → {SCALER_PATH}")
    print(f"\n[DONE] Now run: python room_fall_detector.py")
    print(f"  The detector will use the trained model for fall confirmation.")


if __name__ == "__main__":
    main()
