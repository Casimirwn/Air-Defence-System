#!/usr/bin/env python3
"""Simple servo test on GPIO 12."""

from gpiozero import AngularServo
from time import sleep

# min_pulse_width and max_pulse_width may need tweaking for your specific servo.
# Standard hobby servos (SG90, MG996R, etc.) usually use 0.5ms–2.5ms.
servo = AngularServo(
    12,
    min_angle=-90,
    max_angle=90,
    min_pulse_width=0.5/1000,
    max_pulse_width=2.5/1000,
)

try:
    while True:
        print("Center (0°)")
        servo.angle = 0
        sleep(1)

        print("Min (-90°)")
        servo.angle = -90
        sleep(1)

        print("Center (0°)")
        servo.angle = 0
        sleep(1)

        print("Max (+90°)")
        servo.angle = 90
        sleep(1)

except KeyboardInterrupt:
    print("\nStopping.")
    servo.detach()
