cmake_minimum_required(VERSION 3.0.2)
project(gaussian_splatting)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    nodes/gaussian_splatting_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

