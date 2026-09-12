cmake_minimum_required(VERSION 3.8)
project(chopped_urdf_v2_gazebo)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch worlds rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
