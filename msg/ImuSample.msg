# IMU sample carrying the sensor timestamp as a double, so a replayed dataset
# reproduces the offline result exactly. ROS Time is integer nanoseconds and a
# round trip through it perturbs the recorded time by ~1e-7 s, which is enough
# to change the estimate on the most ill-conditioned sequences.
Header  header
float64 stamp                       # sensor time [s]
geometry_msgs/Vector3 angular_velocity
geometry_msgs/Vector3 linear_acceleration
