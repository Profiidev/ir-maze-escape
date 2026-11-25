from controller import Robot

# Create the Robot instance.
robot = Robot()

# Get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Get joint motors
motor_left  = robot.getDevice('left wheel motor')
motor_right = robot.getDevice('right wheel motor')
# Set positions to infinity for velocity control
motor_left.setPosition(float('inf'))
motor_right.setPosition(float('inf'))

# Get sensors and enable readings
# 1. Distance sensor
distance = robot.getDevice('distance sensor')
distance.enable(timestep)

# 2. Distance sensor
distance2 = robot.getDevice('distance sensor(1)')
distance2.enable(timestep)

# 2. IMU sensor
imu = robot.getDevice('inertial unit')
imu.enable(timestep)

# Mechanical parameters
radius = 0.02
beam   = 0.026 * 2

dist_ref = 0.1
v = 0.05

# YOUR CODE: Control tuning
k_y   = 0.1
k_psi = 0.1

const_rad90 = 1.567
yaw_ref = -1.567

dist_threshold = 0.15

rotation_required = False
# YOUR CODE: Define reference values for distance and yaw-angle
# k_ref_y =
# k_ref_psi =

# Main loop:
# - perform simulation steps until Webots is stopping the controller
t = 0
time_rot = 0

turn = False

while robot.step(timestep) != -1:
    # Read distance and IMU sensors
    psi = imu.getRollPitchYaw()[2]
    dist = distance.getValue()
    dist2 = distance2.getValue()

    # print(psi, dist)

    if(dist2 < dist_ref + dist_threshold):
        if(not turn):
            print("dist2: " + str(dist2))
            
            if(dist > dist_ref + dist_threshold):
                yaw_ref -= const_rad90
                print("Change yaw ref right")

            else:
                yaw_ref += const_rad90
                print("Change yaw ref left")

            turn = True
            time_rot = t

    # else:
        # rotation_required = False
    # elif(t-time_rot > 5):
    #     rotation_required = False
    #     time_rot = 0
    
    d_y = dist - dist_ref

    print(psi, yaw_ref)

    d_psi = psi - yaw_ref
    # Controller formulation 
    w = -k_y * d_y - k_psi * d_psi

    left_speed = 0
    right_speed = 0

    if(abs(d_psi) > 0.4 and turn):
        print("Turn movement: " + str(d_psi))
        left_speed = -1/2 * w * beam / radius
        right_speed = 1/2 * w * beam / radius
    elif(abs(d_psi) < 0.4 and turn):
        turn = False
    else:
        # model equations for the speeds
        left_speed = v/radius - 1/2 * w * beam / radius
        right_speed = v/radius + 1/2 * w * beam / radius
  
    # Send actuator commands as rotational speeds
    motor_left.setVelocity(left_speed)
    motor_right.setVelocity(right_speed)
    t+=1
    
