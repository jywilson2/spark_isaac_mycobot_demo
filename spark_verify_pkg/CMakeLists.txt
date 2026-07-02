# =============================================================================
# ROS 2 FUNDAMENTALS — THE BUILD SYSTEM: ROS 2 packages are built by colcon,
# which reads package.xml for dependencies and delegates the actual build to
# the "build type" declared there. This package uses ament_cmake (CMake with
# ROS extensions), which is required when a package mixes C++ nodes, Python
# nodes, AND generated message interfaces like this one does. Pure-Python
# packages would use ament_python (setup.py only) instead.
#
# Build/test loop from the workspace root:
#   colcon build --packages-select spark_verify_pkg
#   source install/setup.bash
#   colcon test  --packages-select spark_verify_pkg && colcon test-result --all
# =============================================================================
cmake_minimum_required(VERSION 3.8)
project(spark_verify_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find_package pulls in each dependency's exported CMake config. Every
# package listed here must also appear in package.xml so rosdep/colcon can
# resolve it — CMake and package.xml must be kept in sync by hand.
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# MESSAGE GENERATION: rosidl compiles the .msg interface definitions into
# per-language code — C++ headers (spark_verify_pkg/msg/*.hpp) and Python
# classes (spark_verify_pkg.msg.*) — plus the DDS type support that lets
# rclcpp and rclpy nodes exchange these types on the wire. DEPENDENCIES
# lists packages whose types are embedded in ours (std_msgs/Header).
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/NitrosFrameHandle.msg"
  "msg/BlockDetection.msg"
  "msg/RlObservation.msg"
  "msg/SerialCommand.msg"
  "msg/PolicyInference.msg"
  "msg/EdgeHealth.msg"
  DEPENDENCIES std_msgs
)

# LIBRARIES: robot logic (kinematics, safety, encoding) is built as shared
# libraries with no node code, so both the executables and the gtest
# binaries can link the exact same object code that ships.
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

# EXECUTABLES: the C++ nodes. ament_target_dependencies wires up include
# paths and linking for ROS packages found via find_package above.
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

# Linking against OUR OWN generated messages needs this special dance: the
# generated C++ type support is a target created by rosidl at configure
# time, retrieved here and linked explicitly.
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(pymycobot_driver "${cpp_typesupport_target}")

# INSTALL RULES: `ros2 run <pkg> <exe>` and launch files search
# lib/${PROJECT_NAME}/ for executables; libraries go to lib/ proper.
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

# Launch files are DATA, installed to share/; ros2 launch finds them via
# the ament index.
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

# Python node executables: install(PROGRAMS) preserves the executable bit
# on the thin wrapper scripts in scripts/ (see those files for why
# ament_cmake packages need wrappers instead of setup.py entry points).
install(PROGRAMS
  scripts/mock_camera_publisher
  scripts/joint_command_dispatcher
  scripts/block_vision_tracker
  scripts/rl_observation_bridge
  scripts/onnx_inference_node
  scripts/edge_deployment_node
  scripts/mock_usb_camera_hil
  DESTINATION lib/${PROJECT_NAME})

# Installs the spark_verify_nodes/ Python module into site-packages so
# both the wrapper scripts and the tests can `import spark_verify_nodes`.
ament_python_install_package(spark_verify_nodes
  SCRIPTS_DESTINATION lib/${PROJECT_NAME}
)

# TESTING: BUILD_TESTING is colcon's standard switch (on by default,
# disabled with --cmake-args -DBUILD_TESTING=OFF). Everything in this block
# only exists for `colcon test`.
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  find_package(launch_testing_ament_cmake REQUIRED)
  # Auto-registers the linters declared as test_depends in package.xml
  # (flake8, pep257, cpplint, uncrustify, xmllint, ...). Lint failures fail
  # `colcon test` just like functional test failures.
  ament_lint_auto_find_test_dependencies()

  # C++ unit tests: one gtest binary per library under test.
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

  # Python unit tests (plain pytest, no ROS graph). APPEND_ENV puts the
  # package SOURCE dir on PYTHONPATH so `import spark_verify_nodes` works
  # even before the package is installed.
  ament_add_pytest_test(test_camera_lens_advisor test/test_camera_lens_advisor.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60)
  ament_add_pytest_test(test_mock_onnx_policy test/test_mock_onnx_policy.py
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}
    TIMEOUT 60)

  # Integration tests: add_launch_test runs each file under
  # launch_testing, which starts the mock ecosystem processes and executes
  # the embedded unittest cases against the live graph.
  add_launch_test(test/test_phase1_integration.py TIMEOUT 60)
  add_launch_test(test/test_phase2_integration.py TIMEOUT 60)
  add_launch_test(test/test_phase3_integration.py TIMEOUT 60)
  add_launch_test(test/test_phase4_hil_integration.py TIMEOUT 60)
endif()

# Export headers/libraries so a downstream package could link against the
# safety evaluator or encoder with just find_package(spark_verify_pkg).
ament_export_include_directories(include)
ament_export_libraries(articulation_model safety_boundary_evaluator pymycobot_serial_encoder)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
