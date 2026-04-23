import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from rpi_hardware_pwm import HardwarePWM
    SERVO_AVAILABLE = True
except ImportError:
    print('WARNING: rpi-hardware-pwm not found. Install with: pip install rpi-hardware-pwm')
    print('Also make sure hardware PWM is enabled: add "dtoverlay=pwm-2chan" to /boot/firmware/config.txt and reboot.')
    SERVO_AVAILABLE = False

def servo_angle_to_duty(angle):
    """Convert angle (0-180) to duty cycle for a standard servo.
    0 degrees = 2.5% duty, 90 degrees = 7.5%, 180 degrees = 12.5%."""
    return 2.5 + (angle / 180.0) * 10.0

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--pan_pin', help='Hardware PWM channel for pan servo (default: 0 = GPIO12)', type=int, default=0)
parser.add_argument('--tilt_pin', help='Hardware PWM channel for tilt servo (default: 1 = GPIO13)', type=int, default=1)

args = parser.parse_args()


# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record
pan_pin = args.pan_pin
tilt_pin = args.tilt_pin

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

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
    
    # Set up recording
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

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize servo motors using HARDWARE PWM (no jitter)
# Pi 5 has 4 hardware PWM channels. With dtoverlay=pwm-2chan:
#   PWM channel 0 = GPIO 12 (physical pin 32) -> pan servo
#   PWM channel 1 = GPIO 13 (physical pin 33) -> tilt servo
# Servos expect 50Hz PWM. Duty cycle: 2.5% = 0deg, 7.5% = 90deg, 12.5% = 180deg
if SERVO_AVAILABLE:
    pan_pwm = HardwarePWM(pwm_channel=pan_pin, hz=50, chip=2)   # chip=2 for Pi 5
    tilt_pwm = HardwarePWM(pwm_channel=tilt_pin, hz=50, chip=2)
    pan_pwm.start(servo_angle_to_duty(90))   # start centered
    tilt_pwm.start(servo_angle_to_duty(90))
    print(f'Hardware PWM servos initialized: pan=channel {pan_pin}, tilt=channel {tilt_pin}')

# Current servo angles (0-180, 90 = center)
current_pan = 90.0
current_tilt = 90.0

# Servo update timer - only update every ~1 second
servo_update_interval = 1.0  # seconds
last_servo_update = 0.0

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Begin inference loop
while True:

    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder': # If source is image or image folder, load the image using its filename
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video': # If source is a video, load next frame from video file
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb': # If source is a USB camera, grab frame from camera
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera': # If source is a Picamera, grab frames using picamera interface
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Run inference on frame
    results = model(frame, verbose=False, classes=[0])

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):

        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get bounding box confidence
        conf = detections[i].conf.item()

        # Draw box if confidence threshold is high enough
        if conf > float(min_thresh):

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

            # Basic example: count the number of objects in the image
            object_count = object_count + 1

    # --- SERVO TRACKING (updates every ~1 second) ---
    now = time.perf_counter()
    if SERVO_AVAILABLE and (now - last_servo_update) >= servo_update_interval:
        last_servo_update = now

        if len(detections) > 0:
            # Pick the detection with highest confidence
            best_idx = 0
            best_conf = 0
            for i in range(len(detections)):
                c = detections[i].conf.item()
                if c > float(min_thresh) and c > best_conf:
                    best_conf = c
                    best_idx = i

            if best_conf > float(min_thresh):
                xyxy = detections[best_idx].xyxy.cpu().numpy().squeeze().astype(int)
                bx_center = (xyxy[0] + xyxy[2]) / 2.0
                by_center = (xyxy[1] + xyxy[3]) / 2.0

                frame_h, frame_w = frame.shape[:2]

                # Error: how far off-center the object is (-1 to +1)
                error_x = (frame_w / 2.0 - bx_center) / (frame_w / 2.0)
                error_y = (frame_h / 2.0 - by_center) / (frame_h / 2.0)

                # Smooth movement: nudge angle by a fraction of the error
                move_speed = 15.0  # degrees per update (tune this: 5=slow, 30=fast)
                current_pan = max(0, min(180, current_pan + error_x * move_speed))
                current_tilt = max(0, min(180, current_tilt + error_y * move_speed))

                pan_pwm.change_duty_cycle(servo_angle_to_duty(current_pan))
                tilt_pwm.change_duty_cycle(servo_angle_to_duty(current_tilt))
                print(f'Servo update -> pan: {current_pan:.1f}deg, tilt: {current_tilt:.1f}deg')

    # Draw servo position on frame
    if SERVO_AVAILABLE:
        cv2.putText(frame, f'Pan: {current_pan:.0f}deg Tilt: {current_tilt:.0f}deg', (10,60), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    # Calculate and draw framerate (if using video, USB, or Picamera source)
    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw framerate
    
    # Display detection results
    cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects
    cv2.imshow('YOLO detection results',frame) # Display image
    if record: recorder.write(frame)

    # If inferencing on individual images, wait for user keypress before moving to next image. Otherwise, wait 5ms before moving to next frame.
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)
    
    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
if SERVO_AVAILABLE:
    pan_pwm.change_duty_cycle(servo_angle_to_duty(90))  # Return to center
    tilt_pwm.change_duty_cycle(servo_angle_to_duty(90))
    time.sleep(0.5)
    pan_pwm.stop()
    tilt_pwm.stop()
    print('Servos returned to center and stopped.')
cv2.destroyAllWindows()
