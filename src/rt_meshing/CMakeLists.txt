cmake_minimum_required(VERSION 3.0.2)
project(rt_meshing)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  message_filters
  visualization_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    nodes/rt_mesh_cvi_node.py
    nodes/rt_mesh_cvi_update_node.py
    nodes/rt_mesh_update_node.py
    nodes/rt_geometry_points_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
