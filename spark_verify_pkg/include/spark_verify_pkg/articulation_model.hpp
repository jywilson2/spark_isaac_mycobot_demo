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

#ifndef SPARK_VERIFY_PKG__ARTICULATION_MODEL_HPP_
#define SPARK_VERIFY_PKG__ARTICULATION_MODEL_HPP_

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"

namespace spark_verify_pkg
{

constexpr std::size_t kMyCobotDof = 6;

/// Simplified serial-chain kinematics for the MyCobot 280 mock articulation tree.
class ArticulationModel
{
public:
  ArticulationModel();

  void set_joint_positions(
    const std::vector<std::string> & names,
    const std::vector<double> & positions);

  const std::array<double, kMyCobotDof> & joint_positions() const
  {
    return joint_positions_;
  }

  std::vector<geometry_msgs::msg::TransformStamped> compute_link_transforms(
    const std::string & base_frame,
    const std::string & /*stamp_frame*/,
    double stamp_sec,
    uint32_t stamp_nsec) const;

  static std::array<std::string, kMyCobotDof> default_joint_names();

private:
  std::array<double, kMyCobotDof> joint_positions_;
};

}  // namespace spark_verify_pkg

#endif  // SPARK_VERIFY_PKG__ARTICULATION_MODEL_HPP_
