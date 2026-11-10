"""Drive the e-puck along the planned route (customData) using GPS + compass."""

import json
import math

from controller import Robot

WHEEL = 0.0205  # m
AXLE = 0.052  # m
SPEED = 0.09  # m/s cruise
TURN = 0.4  # rad: above this error, rotate in place
REACHED = 0.03  # m

robot = Robot()
dt = int(robot.getBasicTimeStep())
route = json.loads(robot.getCustomData())
gps, compass = robot.getDevice("gps"), robot.getDevice("compass")
gps.enable(dt)
compass.enable(dt)
motors = [robot.getDevice(n) for n in ("left wheel motor", "right wheel motor")]
for m in motors:
    m.setPosition(float("inf"))


def drive(v, w):
    for m, s in zip(motors, (-1, 1)):
        m.setVelocity(max(-6.28, min(6.28, (v + s * w * AXLE / 2) / WHEEL)))


i = 1
while robot.step(dt) != -1 and i < len(route):
    x, y, _ = gps.getValues()
    c = compass.getValues()
    heading = math.atan2(c[0], c[1])
    dx, dy = route[i][0] - x, route[i][1] - y
    if math.hypot(dx, dy) < REACHED:
        i += 1
        continue
    err = (math.atan2(dy, dx) - heading + math.pi) % (2 * math.pi) - math.pi
    if abs(err) > TURN:
        drive(0, 3 * err)
    else:
        drive(SPEED * (1 - abs(err) / TURN), 4 * err)
drive(0, 0)
print("reached the portal")
