#!/usr/bin/env python3
"""Test two servos on GPIO 12 and GPIO 13."""

from gpiozero import AngularServo
from time import sleep

servo1 = AngularServo(
    12,
    min_angle=-90,
    max_angle=90,
    min_pulse_width=0.5/1000,
    max_pulse_width=2.5/1000,
)

servo2 = AngularServo(
    13,
    min_angle=-90,
    max_angle=90,
    min_pulse_width=0.5/1000,
    max_pulse_width=2.5/1000,
)

try:
    while True:
        print("Both center")
        servo1.angle = 0
        servo2.angle = 0
        sleep(1)

        print("Servo1 left, Servo2 right")
        servo1.angle = -90
        servo2.angle = 90
        sleep(1)

        print("Both center")
        servo1.angle = 0
        servo2.angle = 0
        sleep(1)

        print("Servo1 right, Servo2 left")
        servo1.angle = 90
        servo2.angle = -90
        sleep(1)

except KeyboardInterrupt:
    print("\nStopping.")
    servo1.detach()
    servo2.detach()
