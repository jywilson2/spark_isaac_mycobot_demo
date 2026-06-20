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
: joint_positions_{}
{
}

std::array<std::string, kMyCobotDof> ArticulationModel::default_joint_names()
{
  return {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
}

void ArticulationModel::set_joint_positions(
  const std::vector<std::string> & names,
  const std::vector<double> & positions)
{
  if (names.size() != positions.size()) {
    throw std::invalid_argument("Joint names and positions must have equal length");
  }

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
  static constexpr std::array<double, kMyCobotDof> kLinkOffsets = {
    0.1315, 0.1104, 0.0948, 0.0704, 0.0525, 0.0405};

  std::vector<geometry_msgs::msg::TransformStamped> transforms;
  transforms.reserve(kMyCobotDof);

  std::string parent_frame = base_frame;
  for (std::size_t idx = 0; idx < kMyCobotDof; ++idx) {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp.sec = static_cast<int32_t>(stamp_sec);
    tf.header.stamp.nanosec = stamp_nsec;
    tf.header.frame_id = parent_frame;
    tf.child_frame_id = "link" + std::to_string(idx + 1);

    const double angle = joint_positions_[idx];
    tf.transform.translation.x = kLinkOffsets[idx] * std::cos(angle);
    tf.transform.translation.y = kLinkOffsets[idx] * std::sin(angle);
    tf.transform.translation.z = 0.0;
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
