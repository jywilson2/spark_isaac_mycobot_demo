cmake_minimum_required(VERSION 3.8)
project(spark_verify_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/NitrosFrameHandle.msg"
  "msg/BlockDetection.msg"
  "msg/RlObservation.msg"
  "msg/SerialCommand.msg"
  "msg/PolicyInference.msg"
  "msg/EdgeHealth.msg"
  DEPENDENCIES std_msgs
)

add_library(articulation_model SHARED src/articulation_model.cpp)
target_include_directories(articulation_model PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(articulation_model geometry_msgs)

add_library(safety_boundary_evaluator SHARED src/safety_boundary_evaluator.cpp)
target_include_directories(safety_boundary_evaluator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(safety_boundary_evaluator articulation_model)

add_library(pymycobot_serial_encoder SHARED src/pymycobot_serial_encoder.cpp)
target_include_directories(pymycobot_serial_encoder PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(pymycobot_serial_encoder articulation_model)

add_executable(mock_articulation_bridge src/mock_articulation_bridge.cpp)
target_include_directories(mock_articulation_bridge PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mock_articulation_bridge articulation_model)
ament_target_dependencies(
  mock_articulation_bridge
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2_ros)

add_executable(pymycobot_driver src/pymycobot_driver.cpp)
target_include_directories(pymycobot_driver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(pymycobot_driver
  pymycobot_serial_encoder
  safety_boundary_evaluator)
ament_target_dependencies(pymycobot_driver rclcpp)

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(pymycobot_driver "${cpp_typesupport_target}")

install(TARGETS
  articulation_model
  safety_boundary_evaluator
  pymycobot_serial_encoder
  mock_articulation_bridge
  pymycobot_driver
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
  DESTINATION include)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/mock_camera_publisher
  scripts/joint_command_dispatcher
  scripts/block_vision_tracker
  scripts/rl_observation_bridge
  scripts/onnx_inference_node
  scripts/edge_deployment_node
  scripts/mock_usb_camera_hil
  DESTINATION lib/${PROJECT_NAME})

ament_python_install_package(spark_verify_nodes
  SCRIPTS_DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_articulation_model test/test_articulation_model.cpp)
  if(TARGET test_articulation_model)
    target_include_directories(test_articulation_model PUBLIC include)
    target_link_libraries(test_articulation_model articulation_model)
  endif()

  ament_add_gtest(test_safety_boundaries test/test_safety_boundaries.cpp)
  if(TARGET test_safety_boundaries)
    target_include_directories(test_safety_boundaries PUBLIC include)
    target_link_libraries(test_safety_boundaries safety_boundary_evaluator)
  endif()

  ament_add_gtest(test_pymycobot_serial_encoder test/test_pymycobot_serial_encoder.cpp)
  if(TARGET test_pymycobot_serial_encoder)
    target_include_directories(test_pymycobot_serial_encoder PUBLIC include)
    target_link_libraries(test_pymycobot_serial_encoder pymycobot_serial_encoder)
  endif()

  add_launch_test(test/test_phase1_integration.py TIMEOUT 60)
  add_launch_test(test/test_phase2_integration.py TIMEOUT 60)
  add_launch_test(test/test_phase3_integration.py TIMEOUT 60)
  add_launch_test(test/test_phase4_hil_integration.py TIMEOUT 60)
endif()

ament_export_include_directories(include)
ament_export_libraries(articulation_model safety_boundary_evaluator pymycobot_serial_encoder)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
