// Copyright 2026 spark_isaac_mycobot_demo contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ---------------------------------------------------------------------------
// ROS 2 FUNDAMENTALS: This header defines a plain C++ class with NO ROS
// dependencies beyond message types. Keeping robot math in a library that is
// separate from any rclcpp::Node is a core ROS 2 design pattern: the logic
// can be unit-tested with plain gtest (see test/test_articulation_model.cpp)
// without starting a ROS executor, DDS discovery, or any pub/sub machinery.
// The node classes (e.g. mock_articulation_bridge.cpp) then act as thin
// adapters that wire this library into topics, parameters, and TF.
// ---------------------------------------------------------------------------

#ifndef SPARK_VERIFY_PKG__ARTICULATION_MODEL_HPP_
#define SPARK_VERIFY_PKG__ARTICULATION_MODEL_HPP_

#include <array>
#include <cmath>
#include <string>
#include <vector>

// geometry_msgs is one of ROS 2's "common interface" packages. A
// TransformStamped is the standard message for expressing the pose of one
// coordinate frame (child_frame_id) relative to another (header.frame_id)
// at a specific time (header.stamp). The TF2 library consumes streams of
// these messages to build a time-buffered transform tree.
#include "geometry_msgs/msg/transform_stamped.hpp"

namespace spark_verify_pkg
{

// The Elephant Robotics MyCobot 280 is a 6 degree-of-freedom (DOF) serial
// arm: six revolute joints connected base-to-tip. This constant is shared by
// every component in the package (encoder, safety evaluator, driver) so the
// DOF is defined exactly once.
constexpr std::size_t kMyCobotDof = 6;

/// Simplified serial-chain kinematics for the MyCobot 280 mock articulation tree.
///
/// "Articulation" is Isaac Sim / PhysX terminology for a jointed rigid-body
/// tree (what URDF calls a robot model). In the live system this class is
/// replaced by the Isaac Sim Articulation API, which applies joint targets
/// directly on the GPU-simulated robot. The mock keeps only what the tests
/// need: joint positions in, per-link transforms out.
class ArticulationModel
{
public:
  ArticulationModel();

  /// Update stored joint positions from a (names, positions) pair, matching
  /// the layout of sensor_msgs/msg/JointState. ROS convention: JointState
  /// carries parallel arrays, and consumers must match entries BY NAME, not
  /// by index, because publishers may order joints arbitrarily or send a
  /// subset. Throws std::invalid_argument for unknown names or length
  /// mismatch so callers can reject malformed commands loudly.
  void set_joint_positions(
    const std::vector<std::string> & names,
    const std::vector<double> & positions);

  /// Read access to the current joint vector, radians, base-to-tip order.
  const std::array<double, kMyCobotDof> & joint_positions() const
  {
    return joint_positions_;
  }

  /// Forward kinematics (simplified): produce one TransformStamped per link,
  /// chained parent->child (base_link -> link1 -> ... -> link6). Broadcasting
  /// these over the /tf topic lets any node in the graph resolve "where is
  /// link6 relative to base_link?" via a tf2 Buffer lookup.
  std::vector<geometry_msgs::msg::TransformStamped> compute_link_transforms(
    const std::string & base_frame,
    const std::string & /*stamp_frame*/,
    double stamp_sec,
    uint32_t stamp_nsec) const;

  /// Canonical MyCobot joint names ("joint1".."joint6"). These must agree
  /// with the URDF in elephantrobotics/mycobot_ros2 and with the Python
  /// mirror in spark_verify_nodes/articulation_model_py.py.
  static std::array<std::string, kMyCobotDof> default_joint_names();

private:
  std::array<double, kMyCobotDof> joint_positions_;
};

}  // namespace spark_verify_pkg

#endif  // SPARK_VERIFY_PKG__ARTICULATION_MODEL_HPP_
