cmake_minimum_required(VERSION 3.8)
project(bracketbot_moveit_config)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/record_demo.sh
    scripts/bag_to_csv.py
    scripts/play_demo.py
  DESTINATION lib/${PROJECT_NAME}
)

# Lets `moveit_setup_assistant` re-open this package for editing.
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})

ament_package()
