# Group 03 - Air Defense System

## Table of Contents

1. [Project Idea and Goals](#1-project-idea-and-goals)
2. [Problems and Solutions](#2-problems-and-solutions)
   - [Computer Vision Implementation](#21-computer-vision-implementation)
   - [Motor and Aiming Implementation](#22-motor-and-aiming-implementation)
   - [Modeling and Assembly](#23-modeling-and-assembly)
3. [Components](#3-components)
4. [Final Result](#4-final-result)
5. [Code](#5-code)
6. [Wiring Diagram](#6-wiring-diagram)
7. [Photos](#7-photos)
8. [Videos](#8-videos)
9. [Inspirations](#9-inspirations)
10. [Reflections and Development Ideas](#10-reflections-and-development-ideas)

---

## 1. Project Idea and Goals

The goal of the project is to build an automatic targeting system that can identify a target based on camera feed and direct a turret toward it.

The primary objective is to have the turret track a target in real time by combining computer vision and motors. In our view, the system is already a success if the camera can detect a target and the turret turns toward it automatically.

We broke the project down into intermediate milestones:

- **Milestone 1:** Get the camera feed visible on a computer. This took longer than expected due to debugging, but it serves as the foundation for everything else.
- **Milestone 2:** Implement object detection from the camera feed. We researched suitable computer vision solutions such as OpenCV and pre-trained models that allow locating objects within an image.
- **Milestone 3:** Combine the detection with the turret's movement so the system calculates the required aiming angles based on the target's position and drives the motors accordingly.

If we had had significantly more time, we would have expanded the range of motion to 360 degrees. We also considered mounting the air defense system on a remote-controlled car.

Our team's main goal was above all to learn how to combine computer vision and mechanics in practice. The final result was likely to evolve as we encountered challenges and new ideas throughout the project.

---

## 2. Problems and Solutions

### 2.1 Computer Vision Implementation

Computer vision is implemented using the pre-trained **YOLO11n** model, which runs on images captured by a camera module connected to a Raspberry Pi. The model detects objects in real time and returns their positions, which are used to aim the turret.

Setting up the computer vision required significant preparatory work on hardware and software compatibility. We initially used a **Raspberry Pi 4**, where we encountered several issues. Getting the camera feed to display required installing a newer operating system, and loading the YOLO11n model caused issues due to version conflicts with PyTorch and other libraries. Compatible versions were found by trial and error along with some Googling, after which the model worked correctly.

However, the frame rate on the Raspberry Pi 4 was only about **1 frame per second**, which was insufficient for real-time targeting. We therefore switched to a **Raspberry Pi 5**, which brought the frame rate up to approximately **8 frames per second**. Switching to the new device required reinstalling the operating system and all libraries from scratch.

### 2.2 Motor and Aiming Implementation

The turret is aimed using **servo motors** controlled based on the target position provided by computer vision. When the camera detects a target, the required aiming angle is calculated from its position and the motors are moved accordingly.

Two significant issues arose during motor setup:
1. Some of the original motors turned out to be faulty and had to be replaced. Functionality was verified first with a separate test script.
2. When the motor control was combined with the camera detection script, the motors started behaving unexpectedly — moving on their own and not in the intended direction. The cause turned out to be signal interference in the shared script. The solution was to use a separate **PCA9685 servo motor controller**, which made motor control reliable and precise.

### 2.3 Modeling and Assembly

We used an existing model as the base for the final turret: [Thingiverse #4870102](https://www.thingiverse.com/thing:4870102). Fortunately, the 3D model was made in **Onshape** and an editable Onshape version was available, which we could easily modify to suit our needs. From the Onshape version we could see the model's dimensions, which helped us determine which motors we needed. We enlarged the base part of the 3D model and the size of its openings. We did not print all parts, however, as our goal was not to replicate the movement of the original model but instead to have the turret track a target.

For assembly we primarily used glue, tape, and screws. We screwed a servo motor to the base, enabling the turret to move **180 degrees** on the horizontal axis. We then added parts enabling movement on the vertical axis, which cannot reach 180 degrees due to physical constraints. Finally, we added the magazine-barrel assembly, the magazine, and the firing mechanism. The turret can fire a Nerf dart by using a servo motor to push the round into rotating plastic cylinders. As a last step, the camera was attached to the front of the magazine.

---

## 3. Components

| # | Components |
|---|-----------|
| 1 | Servo Motor – 4.8–6 VDC – Standard – Analog – 180° (SER0020) × 2 |
| 2 | Raspberry Pi 5 |
| 3 | Extended ribbon cable |
| 4 | PCA9685 Servo Motor Controller |
| 5 | Battery holder |
| 6 | Switch |
| 7 | DC motor × 2 |
| 8 | Raspberry Pi Camera Module 3 |
| 9 | Servo Motor MG90S |

---

## 4. Final Result

The end result is a fully functional system: a Raspberry Pi-based air defense setup controlled via camera. The system detects targets in real time using the YOLO11n model, automatically aims the turret toward the detected target using servo motors, and is capable of firing a Nerf dart when the **`F`** key is pressed on the keyboard. Computer vision runs on the Raspberry Pi 5, enabling fast enough response to moving targets. Given our starting point, the final result is quite successful and we can be proud of what we achieved.

---

## 5. Code
Scripts in the Tests folder were used to test components.
"Servomotor.py" was used to test a single servomotor.
"Servomotor2.py" was used to test and control 2 servomotors at the same time.
"yolo_detect_track.py" is the original script used to test out the yolo11n camera detection.
"yolo_detect_pca9685.py" was used to test out the pca9685 board with our 2 servmotors while doing camera detection and tracking

THE MAIN SCRIPT
"yolo_track_pca9685.py" is the main script that we use in our finished product. The script uses the rpicam to detect objects with the yolo11n model and then calculates the movement for the servo motors and also moves them accordingly.

On startup the script initializes all the required libraries for the code.
The main loop runs the whole time.

A frame is grabbed from the source and flipped 180° (to compensate for the upside-down camera mount).
YOLO11n runs inference, filtering for class 0 (person) above the confidence threshold.
Bounding boxes and confidence labels are drawn on the frame.
Every time the servo tracking logic runs: it takes the highest-confidence detection, calculates the center of its bounding box, smooths the position over a 5-frame rolling buffer, computes how far off-center the target is (−1 to +1 error), and moves pan/tilt by up to 10° per update. A small deadzone prevents twitching on minor wobbles.
The frame is displayed and keypresses are checked — F fires the trigger servo, S pauses, P saves a screenshot, Q quits.

Cleanup — on quit, the camera is released, servos return to 90°, and the PCA9685 is deinitialized.

We used the following script as a base, which we modified to suit our purposes better (https://github.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/yolo_detect.py).

---

## 6. Wiring Diagram

We have two separate wiring configurations.

**Main configuration (servo motors + Raspberry Pi):**  
In practice we used rechargeable 1.2 V batteries. The external voltage source is therefore 6 × 1.2 V = **7.2 V**. The capacitor is **100 µF**. The PCA9685 and Raspberry Pi 5 communicate over an **I2C bus**.

**Secondary configuration (DC motors):**  
A switch is used to turn the DC motors on and off. We also used rechargeable 1.2 V batteries as the power source here.

---

## 7. Photos

*(Photos included in the original project document)*

---

## 8. Videos

*(Videos included in the original project document)*

---

## 9. Inspirations
We used a ready made script as a base, which we modified. Additionally we used it as a guide for downloading the libraries: https://www.youtube.com/watch?v=z70ZrSZNi-8

In earlier iterations of the project we used a different 3D model focused on getting the camera working rather than firing: [Thingiverse #4710301](https://www.thingiverse.com/thing:4710301).

The 3D modeling was inspired by an existing motorized Nerf turret model: [Thingiverse #4870102](https://www.thingiverse.com/thing:4870102/files).

---

## 10. Reflections and Development Ideas

None of our team members had previously worked with computer vision or electronics, so almost everything was new to us. As a result, we learned an enormous amount during the course — especially about managing hardware/software compatibility and solving problems in practical situations. For example, we had to swap out the entire hardware platform mid-project and rebuild the environment from scratch, which taught us a great deal about the importance of understanding hardware limitations before building on top of them.

The end result may not be technically cutting-edge, but given where we started and the challenges we faced along the way, we are satisfied with what we accomplished. The computer vision works, the motors move, and the system responds to targets — and that was our goal.

**Ideas we considered but didn't have time to implement:**

- Distance estimation from the camera image, allowing the turret to adjust aiming in three dimensions
- Full 360-degree rotation
- Improved computer vision to predict the movement direction of a target, enabling more accurate tracking of moving objects
- Optimized model performance on the Raspberry Pi 5, or switching to a lighter model specifically designed for edge devices

There were plenty more ideas. This kind of system is easy to extend step by step, and many of the ideas listed above would likely have been achievable within the course timeframe with more time available. Since this was only one course, we had to prioritize and accept that we couldn't do everything. We are still satisfied with what we managed to build.
