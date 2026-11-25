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
# 2. IMU sensor
imu = robot.getDevice('inertial unit')
imu.enable(timestep)

# Mechanical parameters
radius = 0.02
beam   = 0.026 * 2

dist_ref = 0.3
v = 0.05

# YOUR CODE: Control tuning
k_y   = 0.1
k_psi = 0.1

# YOUR CODE: Define reference values for distance and yaw-angle
# k_ref_y =
# k_ref_psi =

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    
    # Read distance and IMU sensors
    psi = imu.getRollPitchYaw()[2]
    dist = distance.getValue()
    print(psi, dist)

    d_y = dist - dist_ref
    d_psi = psi - 0
    # Controller formulation 
    w = -k_y * d_y - k_psi * d_psi

    # model equations for the speeds
    left_speed = v/radius - 1/2 * w * beam / radius
    right_speed = v/radius + 1/2 * w * beam / radius
  
    # Send actuator commands as rotational speeds
    motor_left.setVelocity(left_speed)
    motor_right.setVelocity(right_speed)
    
