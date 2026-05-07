# 🛡️ Smart Fall Detection System (Pi5 + Android)

A real-time, AI-powered fall detection system that combines edge computing on a Raspberry Pi 5 with an instant Android alert application.

---

## 🏗️ System Architecture

1.  **Detection Layer (Edge)**: A Raspberry Pi 5 equipped with a Hailo-8L AI HAT (13 TOPS) runs high-speed YOLOv8-pose estimation.
2.  **Communication Layer**: Alerts are dispatched via the [ntfy.sh](https://ntfy.sh) pub-sub service.
3.  **Alert Layer (Mobile)**: An Android background service listens for events and triggers a loud, persistent alarm (ringtone + vibration) on the caregiver's device.

---

## 🐍 1. Fall Detection (Raspberry Pi 5)

The core logic uses GStreamer and the Hailo-apps framework to perform accelerated inference.

### 📋 Requirements
- **Hardware**: Raspberry Pi 5 + Hailo AI HAT.
- **Software**: 
    - [Hailo Software Suite](https://hailo.ai/developer-zone/software-downloads/) installed.
    - `hailo-apps` repository installed on the Pi.

### 🚀 How to Run
1. Clone the repository to your Pi.
2. Install dependencies:
   ```bash
   pip install -r requirements_pi.txt
   ```
3. Run the detector:
   ```bash
   # Use --input rpi for the official camera or /dev/video0 for USB
   python room_fall_detector_pi.py --input rpi
   ```

### 🧠 How the Model Works
- **Model**: YOLOv8-pose (Pose Estimation).
- **Heuristics**:
    - **Bounding Box Ratio**: Detects when a person's width exceeds their height significantly (horizontal orientation).
    - **Keypoint Velocity**: Monitors the speed of head/shoulder descent.
    - **Ground Proximity**: Analyzes the distance of keypoints from the estimated floor plane.
    - **Persistence**: A "Fall" is only triggered if the horizontal state persists for a defined number of frames to reduce false positives.

---

## 📱 2. Android Alert App

The companion app ensures that caregivers never miss a fall event, even if the phone is locked.

### 📥 Installation
- Locate the `VibrationApp.apk` in the `app/build/outputs/apk/debug/` directory (or download it from your phone's Documents/Downloads if transferred).
- Install the APK on any Android device (v7.0+).

### ⚙️ Features
- **Instant Alerts**: Uses Server-Sent Events (SSE) for low-latency notifications.
- **Persistent Alarm**: Triggers a continuous ringtone and vibration.
- **Background Service**: Runs as a Foreground Service to prevent being killed by Android's battery optimization.
- **WakeLock**: Temporarily wakes the CPU to ensure the alarm sounds immediately.

---

## 📡 3. Communication (ntfy.sh)

The Pi and Android app communicate via a shared topic.
- **Default Topic**: `melroy-fall-detector`
- **Privacy**: You can change the `NTFY_TOPIC` in `room_fall_detector_pi.py` and the corresponding topic in the Android source code to your own private name for security.

---

## 🛠️ Project Structure
```text
.
├── app/                        # Android App Source Code
├── room_fall_detector_pi.py    # Python Inference Script (Pi 5)
├── start_fall_detector.sh      # Launcher Script for Pi
├── fall-detector.service       # Systemd service for auto-start
├── requirements_pi.txt         # Pi Python dependencies
└── alarm.wav                   # Local audio alert for Pi (Optional)
```

## 🤝 Credits
Developed by **Melroy Quadros** for real-time safety monitoring.
