cmake_minimum_required(VERSION 2.8.3)
project(pose_utils)

## --- Dependencies ---
find_package(catkin REQUIRED COMPONENTS
  roscpp
)
find_package(Armadillo REQUIRED)

## --- Include directories ---
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${ARMADILLO_INCLUDE_DIRS}
)

## --- Declare catkin package ---
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES pose_utils
  CATKIN_DEPENDS roscpp            # <- add this
  DEPENDS Armadillo                # <- external, not catkin
)


## --- Library target ---
add_library(pose_utils
  src/pose_utils.cpp
)

## --- Link libraries ---
target_link_libraries(pose_utils
  ${catkin_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
)

