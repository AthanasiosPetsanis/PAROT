cmake_minimum_required(VERSION 3.0.2)
project(controller)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  quadrotor_msgs
  visualization_msgs
  message_filters
  tf
  tf2_ros
  tf2_geometry_msgs
  cv_bridge
  mavros_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    nav_msgs
    geometry_msgs
    message_filters
    cv_bridge
    quadrotor_msgs
    visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  nodes/controller_v01.py nodes/pos_cmd_to_local_goal.py nodes/keyboard_teleop_pos_cmd.py
  nodes/viewpoint_visualization_node.py
  nodes/visualize_iros_run_rviz.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
