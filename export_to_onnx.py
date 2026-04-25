"""
Export YOLOv8 Pose Model to ONNX for Hailo AI HAT Deployment
=============================================================
Run on Windows PC. Output .onnx file can be compiled to .hef
via the Hailo Dataflow Compiler (DFC) on an x86_64 Ubuntu machine.

RECOMMENDED: Use the pre-compiled HEF from the Hailo Model Zoo instead.
  The hailo-apps repo auto-downloads yolov8s_pose for Hailo-8L (13 TOPS)
  when you run:  hailo-pose --input rpi

Usage (only if custom model needed):
  python export_to_onnx.py
"""
import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def export_model(name, imgsz=640):
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        print(f"[SKIP] {name} not found")
        return None
    print(f"\nExporting {name} → ONNX...")
    model = YOLO(path)
    out = model.export(format="onnx", imgsz=imgsz, opset=11,
                       simplify=True, dynamic=False, half=False)
    print(f"  [OK] → {out}")
    return out


def main():
    print("=" * 55)
    print("  YOLOv8 → ONNX Exporter (for Hailo AI HAT)")
    print("=" * 55)
    print()
    print("  NOTE: Pre-compiled HEFs are recommended for Hailo-8L.")
    print("  The hailo-apps repo auto-downloads the correct HEF.")
    print("  Only use this script if you need a custom model.")
    print()

    # Export pose model (used for fall detection)
    p = export_model("yolov8n-pose.pt")

    print("\nDone!")
    print("Next steps:")
    print("  Option A (Recommended): Use pre-compiled HEF from Hailo Model Zoo")
    print("    → hailo-apps auto-downloads yolov8s_pose for Hailo-8L")
    print("  Option B (Custom): Compile .onnx → .hef with Hailo DFC")
    print("    → hailo optimize yolov8n-pose.onnx --hw-arch hailo8l")
    print("    → hailo compile yolov8n-pose_optimized.har --hw-arch hailo8l")
    if p:
        print(f"\n  Pose ONNX: {p}")


if __name__ == "__main__":
    main()
