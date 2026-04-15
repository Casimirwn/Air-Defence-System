import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# --- Servo control (Raspberry Pi 5) ---------------------------------------
# Uses gpiozero with the lgpio backend (default on Pi 5).
# GPIO 12 and 13 are hardware-PWM capable -> low jitter.
from gpiozero import AngularServo
from gpiozero.pins.lgpio import LGPIOFactory

PAN_PIN  = 12   # BCM numbering
TILT_PIN = 13

# Servo travel limits (deg). Adjust to your mechanical setup so the servo
# never tries to push past the rig.
PAN_MIN,  PAN_MAX  = -80, 80
TILT_MIN, TILT_MAX = -40, 40

# Starting (centered) angles
pan_angle  = 0.0
tilt_angle = 0.0

# Most hobby servos (SG90, MG90, MG996R) use 0.5–2.5 ms pulses.
factory = LGPIOFactory()
pan_servo = AngularServo(
    PAN_PIN, min_angle=-90, max_angle=90,
    min_pulse_width=0.0005, max_pulse_width=0.0025,
    pin_factory=factory,
)
tilt_servo = AngularServo(
    TILT_PIN, min_angle=-90, max_angle=90,
    min_pulse_width=0.0005, max_pulse_width=0.0025,
    pin_factory=factory,
)

pan_servo.angle  = pan_angle
tilt_servo.angle = tilt_angle

# --- Tracking tuning -------------------------------------------------------
# Proportional gain: degrees of servo movement per pixel of error per frame.
# Lower = smoother but slower; higher = snappier but more overshoot/jitter.
KP_PAN  = 0.03
KP_TILT = 0.03

# Deadzone in pixels: if the target is within this many px of center, do not move.
DEADZONE_PX = 25

# Max degrees the servo is allowed to move in a single frame (rate limit).
MAX_STEP_DEG = 4.0

# If True, invert axis (depends on how your servo is mounted).
INVERT_PAN  = False
INVERT_TILT = False

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def update_servos(target_cx, target_cy, frame_w, frame_h):
    """Move servos to reduce the pixel error between target and frame center."""
    global pan_angle, tilt_angle

    err_x = target_cx - frame_w / 2.0   # +ve => target is right of center
    err_y = target_cy - frame_h / 2.0   # +ve => target is below center

    # Pan: if target is to the right, we typically need to pan right (camera turns right).
    # Tilt: if target is below center, camera should tilt down. Sign depends on mounting.
    dpan  = 0.0 if abs(err_x) < DEADZONE_PX else KP_PAN  * err_x
    dtilt = 0.0 if abs(err_y) < DEADZONE_PX else KP_TILT * err_y

    if INVERT_PAN:  dpan  = -dpan
    if INVERT_TILT: dtilt = -dtilt

    # Rate limit
    dpan  = clamp(dpan,  -MAX_STEP_DEG, MAX_STEP_DEG)
    dtilt = clamp(dtilt, -MAX_STEP_DEG, MAX_STEP_DEG)

    pan_angle  = clamp(pan_angle  + dpan,  PAN_MIN,  PAN_MAX)
    tilt_angle = clamp(tilt_angle + dtilt, TILT_MIN, TILT_MAX)

    pan_servo.angle  = pan_angle
    tilt_servo.angle = tilt_angle

# --- Argument parsing (unchanged from your original) ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True,
                    help='Path to YOLO model file (e.g. "yolo11n.pt")')
parser.add_argument('--source', required=True,
                    help='Image source: file, folder, video, "usb0", or "picamera0"')
parser.add_argument('--thresh', default=0.5, type=float,
                    help='Confidence threshold (default 0.5)')
parser.add_argument('--resolution', default=None,
                    help='Display/inference resolution WxH, e.g. 640x480')
parser.add_argument('--record', action='store_true',
                    help='Record output as demo1.avi (requires --resolution)')
parser.add_argument('--no-servo', action='store_true',
                    help='Disable servo control (useful for testing on a non-Pi machine)')
args = parser.parse_args()

model_path  = args.model
img_source  = args.source
min_thresh  = float(args.thresh)
user_res    = args.resolution
record      = args.record
servo_on    = not args.no_servo

if not os.path.exists(model_path):
    print('ERROR: Model file not found.')
    sys.exit(0)

model  = YOLO(model_path, task='detect')
labels = model.names

img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:   source_type = 'image'
    elif ext in vid_ext_list: source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.'); sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'; usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'; picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid.'); sys.exit(0)

resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

if record:
    if source_type not in ['video','usb','picamera']:
        print('Recording only works for video/camera sources.'); sys.exit(0)
    if not user_res:
        print('Specify --resolution to record.'); sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi',
                               cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# --- Source setup ---------------------------------------------------------
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*')
                 if os.path.splitext(f)[1] in img_ext_list]
elif source_type in ('video', 'usb'):
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW); cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(
        main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

bbox_colors = [(164,120,87),(68,148,228),(93,97,209),(178,182,133),(88,159,106),
               (96,202,231),(159,124,168),(169,162,241),(98,118,150),(172,176,184)]

avg_frame_rate    = 0
frame_rate_buffer = []
fps_avg_len       = 200
img_count         = 0

# --- Main loop ------------------------------------------------------------
try:
    while True:
        t_start = time.perf_counter()

        if source_type in ('image', 'folder'):
            if img_count >= len(imgs_list):
                print('All images processed. Exiting.'); break
            frame = cv2.imread(imgs_list[img_count]); img_count += 1
        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret: print('End of video.'); break
        elif source_type == 'usb':
            ret, frame = cap.read()
            if frame is None or not ret:
                print('Camera read failed.'); break
        elif source_type == 'picamera':
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
            if frame is None: print('Picamera read failed.'); break

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        frame_h, frame_w = frame.shape[:2]

        # Run YOLO -- class 0 = "person" in COCO
        results    = model(frame, verbose=False, classes=[0])
        detections = results[0].boxes

        # Pick the best target: largest bbox above threshold
        best_box  = None
        best_area = 0
        object_count = 0

        for i in range(len(detections)):
            conf = detections[i].conf.item()
            if conf < min_thresh:
                continue

            xyxy = detections[i].xyxy.cpu().numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            classidx  = int(detections[i].cls.item())
            classname = labels[classidx]

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            label = f'{classname}: {int(conf*100)}%'
            (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, lh + 10)
            cv2.rectangle(frame, (xmin, label_ymin - lh - 10),
                          (xmin + lw, label_ymin + bl - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            object_count += 1

            area = (xmax - xmin) * (ymax - ymin)
            if area > best_area:
                best_area = area
                best_box  = (xmin, ymin, xmax, ymax)

        # Drive servos toward the chosen target
        if servo_on and best_box is not None:
            xmin, ymin, xmax, ymax = best_box
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            update_servos(cx, cy, frame_w, frame_h)

            # Visual aids: target center + frame center crosshair
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)
            cv2.drawMarker(frame, (frame_w // 2, frame_h // 2),
                           (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        # HUD
        if source_type in ('video', 'usb', 'picamera'):
            cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'People: {object_count}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Pan:{pan_angle:+.1f}  Tilt:{tilt_angle:+.1f}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('YOLO tracking', frame)
        if record: recorder.write(frame)

        key = cv2.waitKey(5) if source_type in ('video','usb','picamera') else cv2.waitKey()
        if key in (ord('q'), ord('Q')): break
        elif key in (ord('s'), ord('S')): cv2.waitKey()
        elif key in (ord('p'), ord('P')): cv2.imwrite('capture.png', frame)
        elif key in (ord('c'), ord('C')):           # 'c' = recenter servos
            pan_angle = 0.0; tilt_angle = 0.0
            pan_servo.angle = 0; tilt_servo.angle = 0

        t_stop = time.perf_counter()
        fps_now = 1.0 / (t_stop - t_start)
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(fps_now)
        avg_frame_rate = float(np.mean(frame_rate_buffer))

finally:
    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    if source_type in ('video','usb'): cap.release()
    elif source_type == 'picamera':    cap.stop()
    if record: recorder.release()
    cv2.destroyAllWindows()
    # Park servos at center and release
    try:
        pan_servo.angle = 0; tilt_servo.angle = 0
        time.sleep(0.3)
        pan_servo.detach(); tilt_servo.detach()
    except Exception:
        pass
