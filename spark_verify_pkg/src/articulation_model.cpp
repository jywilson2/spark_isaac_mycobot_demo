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

#include "spark_verify_pkg/articulation_model.hpp"

#include <stdexcept>

namespace spark_verify_pkg
{

ArticulationModel::ArticulationModel()
: joint_positions_{}  // value-initialization zeroes all six joints
{
}

std::array<std::string, kMyCobotDof> ArticulationModel::default_joint_names()
{
  // These names match the MyCobot URDF joint names so that a live
  // ros2_control / Isaac Sim bridge can be dropped in without renaming.
  return {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
}

void ArticulationModel::set_joint_positions(
  const std::vector<std::string> & names,
  const std::vector<double> & positions)
{
  // sensor_msgs/JointState is a set of parallel arrays; a message where
  // name.size() != position.size() is malformed and must be rejected.
  if (names.size() != positions.size()) {
    throw std::invalid_argument("Joint names and positions must have equal length");
  }

  // Match incoming joints BY NAME. This is the required way to consume
  // JointState: publishers are free to reorder joints or command a subset,
  // so indexing positions[] by array position would silently move the
  // wrong motor. O(n^2) scan is fine for 6 joints.
  const auto defaults = default_joint_names();
  for (std::size_t i = 0; i < names.size(); ++i) {
    bool matched = false;
    for (std::size_t joint_idx = 0; joint_idx < kMyCobotDof; ++joint_idx) {
      if (names[i] == defaults[joint_idx]) {
        joint_positions_[joint_idx] = positions[i];
        matched = true;
        break;
      }
    }
    if (!matched) {
      // Throwing (rather than ignoring) surfaces typos in launch files or
      // publishers immediately; the calling node catches this and logs it.
      throw std::invalid_argument("Unknown joint name: " + names[i]);
    }
  }
}

std::vector<geometry_msgs::msg::TransformStamped> ArticulationModel::compute_link_transforms(
  const std::string & base_frame,
  const std::string & /*stamp_frame*/,
  double stamp_sec,
  uint32_t stamp_nsec) const
{
  // Approximate link offsets (meters) along the MyCobot serial chain.
  // A real forward-kinematics implementation would use the full URDF
  // (per-joint axes and 3D origins). This mock collapses each link to a
  // planar offset rotated about Z, which is enough for the integration
  // test to verify that commands flow through to TF.
  static constexpr std::array<double, kMyCobotDof> kLinkOffsets = {
    0.1315, 0.1104, 0.0948, 0.0704, 0.0525, 0.0405};

  std::vector<geometry_msgs::msg::TransformStamped> transforms;
  transforms.reserve(kMyCobotDof);

  // Build the chain base_link -> link1 -> link2 -> ... Each transform's
  // header.frame_id is the PARENT frame and child_frame_id is the link
  // itself; tf2 stitches consecutive edges into a tree, so a consumer can
  // ask for base_link->link6 and tf2 will compose the intermediate edges.
  std::string parent_frame = base_frame;
  for (std::size_t idx = 0; idx < kMyCobotDof; ++idx) {
    geometry_msgs::msg::TransformStamped tf;
    // Every stamped ROS message carries builtin_interfaces/Time split into
    // int32 seconds + uint32 nanoseconds. TF lookups are time-indexed, so
    // an accurate stamp matters: tf2 interpolates between buffered samples.
    tf.header.stamp.sec = static_cast<int32_t>(stamp_sec);
    tf.header.stamp.nanosec = stamp_nsec;
    tf.header.frame_id = parent_frame;
    tf.child_frame_id = "link" + std::to_string(idx + 1);

    // Planar revolute joint: translate along the link length rotated by the
    // joint angle, and rotate about Z by the same angle.
    const double angle = joint_positions_[idx];
    tf.transform.translation.x = kLinkOffsets[idx] * std::cos(angle);
    tf.transform.translation.y = kLinkOffsets[idx] * std::sin(angle);
    tf.transform.translation.z = 0.0;
    // Orientation is a unit quaternion. For a pure rotation of `angle`
    // about the Z axis: q = (0, 0, sin(a/2), cos(a/2)). ROS uses (x,y,z,w)
    // ordering with w last.
    tf.transform.rotation.x = 0.0;
    tf.transform.rotation.y = 0.0;
    tf.transform.rotation.z = std::sin(angle / 2.0);
    tf.transform.rotation.w = std::cos(angle / 2.0);

    transforms.push_back(tf);
    parent_frame = tf.child_frame_id;
  }

  return transforms;
}

}  // namespace spark_verify_pkg
