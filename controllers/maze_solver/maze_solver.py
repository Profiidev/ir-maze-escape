"""Solve the maze from sensors only: range rays, wheel encoders, compass and camera.

Depth-first exploration on the 0.4 m cell lattice (cells numbered relative to the start). Each cell's four
sides are probed with three parallel range rays; the robot only drives into cells it has not visited and
backtracks at dead ends. It stops when the camera sees the portal beacon while facing the stairs.
"""

import math

from controller import Robot

CELL = 0.4  # m, lattice pitch
WHEEL = 0.0205  # m
AXLE = 0.052  # m
SPEED = 0.15  # m/s, e-puck v2 wheel limit is 7.536 rad/s = 0.154 m/s
OPEN = 0.40  # m: all three rays longer than this = the next cell is free
BEACON_PIXELS = 60

robot = Robot()
dt = int(robot.getBasicTimeStep())
compass = robot.getDevice("compass")
camera = robot.getDevice("camera")
encoders = [robot.getDevice(n) for n in ("left wheel sensor", "right wheel sensor")]
motors = [robot.getDevice(n) for n in ("left wheel motor", "right wheel motor")]
rays = {(k, j): robot.getDevice(f"d{k}_{j}") for k in range(4) for j in range(3)}
for dev in [compass, camera, *encoders, *rays.values()]:
    dev.enable(dt)
for m in motors:
    m.setPosition(float("inf"))


def tick():
    if robot.step(dt) == -1:
        raise SystemExit


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def heading():
    c = compass.getValues()
    return math.atan2(c[0], c[1])


def drive(v, w):
    for m, s in zip(motors, (-1, 1)):
        m.setVelocity(max(-7.536, min(7.536, (v + s * w * AXLE / 2) / WHEEL)))


def odometer():
    return sum(e.getValue() for e in encoders) * WHEEL / 2


def ray(d, h, j=None):
    """Range in absolute direction d (0=E, 1=N, 2=W, 3=S) while facing h; min over the three rays unless j given."""
    k = (d - h) % 4
    return rays[k, j].getValue() if j is not None else min(rays[k, i].getValue() for i in range(3))


def turn_to(h):
    while abs(err := wrap(h * math.pi / 2 - heading())) > 0.015:
        w = math.copysign(max(0.5, min(3.0, 3 * abs(err))), err)
        drive(0, w)
        tick()
    drive(0, 0)


def move(dist, h):
    """Drive dist metres (negative = reverse) holding heading h and centring between side walls.
    Returns False when something blocks the way."""
    start, t0 = odometer(), robot.getTime()
    sign = math.copysign(1, dist)
    while (s := (odometer() - start) * sign) < abs(dist):
        if sign > 0 and ray(h, h) < 0.065 or robot.getTime() - t0 > 4 * abs(dist) / SPEED + 4:
            drive(0, 0)
            move(-(odometer() - start), h) if sign > 0 else None
            return False
        left, right = ray((h + 1) % 4, h, 0), ray((h + 3) % 4, h, 0)
        lat = (left - right) / 2 if left < 0.3 and right < 0.3 else 0
        drive(sign * SPEED * min(1, 0.3 + (abs(dist) - s) * 8), sign * (2.5 * wrap(h * math.pi / 2 - heading()) + 6 * lat))
        tick()
    drive(0, 0)
    return True


def step_cell(h):
    turn_to(h)
    if not move(CELL, h):
        return False
    if 0.10 < (f := ray(h, h, 0)) < 0.20:  # a wall just ahead: re-sync along the corridor
        move(f - 0.12, h)
    return True


def sees_portal():
    img = camera.getImage()
    n = sum(1 for i in range(0, len(img), 4) if img[i + 1] > 180 and img[i + 2] < 110 and img[i + 1] - img[i] > 60)
    return n >= BEACON_PIXELS and ray(h, h, 0) < 0.30


DX = [(1, 0), (0, 1), (-1, 0), (0, -1)]
for _ in range(10):  # let the sensors settle
    tick()
pos, h = (0, 0), round(heading() / (math.pi / 2)) % 4
visited, blocked, back = {pos}, set(), []
while True:
    if sees_portal():
        while ray(h, h, 0) > 0.07:
            drive(SPEED, 0)
            tick()
        drive(0, 0)
        print(f"reached the portal after exploring {len(visited)} cells")
        break
    free = [d for d in (h, (h + 1) % 4, (h + 3) % 4, (h + 2) % 4)
            if ray(d, h) > OPEN and (pos, d) not in blocked and (pos[0] + DX[d][0], pos[1] + DX[d][1]) not in visited]
    if free:
        d = free[0]
        if step_cell(d):
            pos, h = (pos[0] + DX[d][0], pos[1] + DX[d][1]), d
            visited.add(pos)
            back.append(d)
            if len(visited) % 100 == 0:
                print(f"explored {len(visited)} cells")
        else:
            blocked.add((pos, d))
    elif back:
        d = (back.pop() + 2) % 4
        step_cell(d)
        pos, h = (pos[0] + DX[d][0], pos[1] + DX[d][1]), d
    else:
        print("no way to the portal")
        break
