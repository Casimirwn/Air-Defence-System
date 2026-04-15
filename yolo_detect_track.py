import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of USB camera ("usb0"), \
                    or Picamera ("picamera0")',
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5, type=float)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

# --- Servo tracking arguments ---
parser.add_argument('--track', help='Class name to follow with servos (e.g. "person"). Omit to disable tracking.',
                    default=None)
parser.add_argument('--pan-pin', help='GPIO (BCM) pin connected to the pan servo signal wire.',
                    type=int, default=12)
parser.add_argument('--tilt-pin', help='GPIO (BCM) pin connected to the tilt servo signal wire.',
                    type=int, default=13)
parser.add_argument('--kp', help='Proportional gain for servo control. Lower = smoother but slower.',
                    type=float, default=0.03)
parser.add_argument('--deadzone', help='Pixel error below which servos do not move (prevents jitter).',
                    type=int, default=20)
parser.add_argument('--invert-pan', help='Flip pan direction if servo moves the wrong way.',
                    action='store_true')
parser.add_argument('--invert-tilt', help='Flip tilt direction if servo moves the wrong way.',
                    action='store_true')

args = parser.parse_args()


# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labelmap
model = YOLO(model_path, task='detect')
labels = model.names

# --- Servo setup ---
track_name = args.track
track_classidx = None
pan = None
tilt = None
pan_angle = 0.0
tilt_angle = 0.0

if track_name:
    name_to_idx = {v: k for k, v in labels.items()}
    if track_name not in name_to_idx:
        print(f'ERROR: class "{track_name}" not in model. Available classes: {list(labels.values())}')
        sys.exit(0)
    track_classidx = name_to_idx[track_name]

    try:
        from gpiozero import AngularServo
        from gpiozero.pins.lgpio import LGPIOFactory
    except ImportError:
        print('ERROR: gpiozero / lgpio not installed. Run: sudo apt install python3-gpiozero python3-lgpio')
        sys.exit(0)

    factory = LGPIOFactory()
    # Pulse widths 0.5ms - 2.5ms cover the full range of most hobby servos.
    # If your servo buzzes at the extremes, narrow these (e.g. 0.001 - 0.002).
    pan = AngularServo(args.pan_pin, min_angle=-90, max_angle=90,
                       min_pulse_width=0.0005, max_pulse_width=0.0025,
                       pin_factory=factory)
    tilt = AngularServo(args.tilt_pin, min_angle=-45, max_angle=45,
                        min_pulse_width=0.0005, max_pulse_width=0.0025,
                        pin_factory=factory)
    pan.angle = 0
    tilt.angle = 0
    print(f'Servo tracking enabled for class "{track_name}" (idx {track_classidx}).')
    print(f'  Pan pin: GPIO{args.pan_pin}, Tilt pin: GPIO{args.tilt_pin}, Kp: {args.kp}, Deadzone: {args.deadzone}px')

KP = args.kp
DEADZONE = args.deadzone
PAN_SIGN = -1 if args.invert_pan else 1
TILT_SIGN = -1 if args.invert_tilt else 1

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)

    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors (Tableau 10)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Begin inference loop
while True:

    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            break
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1

    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break

    elif source_type == 'usb':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    object_count = 0
    best_target = None      # (cx, cy) of largest matching detection
    best_area = 0

    # Go through each detection
    for i in range(len(detections)):

        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            object_count = object_count + 1

            # Pick the largest matching detection as the servo target
            if track_classidx is not None and classidx == track_classidx:
                area = (xmax - xmin) * (ymax - ymin)
                if area > best_area:
                    best_area = area
                    best_target = ((xmin + xmax) // 2, (ymin + ymax) // 2)

    # --- Servo update ---
    if pan is not None:
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # Always draw a center crosshair when tracking is on
        cv2.line(frame, (cx, 0), (cx, h), (50, 50, 50), 1)
        cv2.line(frame, (0, cy), (w, cy), (50, 50, 50), 1)

        if best_target is not None:
            err_x = best_target[0] - cx
            err_y = best_target[1] - cy

            if abs(err_x) > DEADZONE:
                pan_angle += PAN_SIGN * (-KP * err_x)
                pan_angle = max(-90, min(90, pan_angle))
                pan.angle = pan_angle
            if abs(err_y) > DEADZONE:
                tilt_angle += TILT_SIGN * (KP * err_y)
                tilt_angle = max(-45, min(45, tilt_angle))
                tilt.angle = tilt_angle

            # Highlight target and draw a line from frame center to it
            cv2.circle(frame, best_target, 10, (0, 255, 0), 2)
            cv2.line(frame, (cx, cy), best_target, (0, 255, 0), 1)

        # HUD: show current servo angles
        cv2.putText(frame, f'Pan: {pan_angle:+.1f}  Tilt: {tilt_angle:+.1f}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)

    # Calculate and draw framerate
    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
    cv2.imshow('YOLO detection results',frame)
    if record: recorder.write(frame)

    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)

    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture.png',frame)
    elif key == ord('c') or key == ord('C'):
        # Recenter servos
        if pan is not None:
            pan_angle = 0.0
            tilt_angle = 0.0
            pan.angle = 0
            tilt.angle = 0

    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
if pan is not None:
    pan.detach()
    tilt.detach()
cv2.destroyAllWindows()
