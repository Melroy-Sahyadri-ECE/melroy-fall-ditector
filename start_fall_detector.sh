#!/bin/bash
# Start the Fall Detector with all required environment variables
export DISPLAY=:0
export QT_QPA_PLATFORM=xcb
export GST_PLUGIN_FEATURE_RANK="vaapidecodebin:NONE"
export PYTHONPATH="/home/tce/tce/hailo-rpi5-examples/hailo-apps:$PYTHONPATH"
export HAILO_ENV_FILE="/home/tce/tce/hailo-rpi5-examples/.env"
export TAPPAS_POST_PROC_DIR="/usr/lib/aarch64-linux-gnu/hailo/tappas/post_processes"
export XAUTHORITY=/home/tce/.Xauthority

cd /home/tce/tce/hailo-rpi5-examples/hailo-apps

exec /home/tce/tce/hailo-rpi5-examples/venv_hailo_rpi_examples/bin/python \
    /home/tce/tce/melroy-fall-ditector/room_fall_detector_pi.py \
    --input usb --width 640 --height 480
