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

#ifndef SPARK_VERIFY_PKG__SAFETY_BOUNDARY_EVALUATOR_HPP_
#define SPARK_VERIFY_PKG__SAFETY_BOUNDARY_EVALUATOR_HPP_

#include <array>
#include <cmath>

#include "spark_verify_pkg/articulation_model.hpp"

namespace spark_verify_pkg
{

struct SafetyBoundaryConfig
{
  double max_joint_velocity_rad_s = 2.0;
  std::array<double, kMyCobotDof> joint_min = {
    -2.879793, -1.570796, -2.879793, -2.879793, -2.879793, -2.879793};
  std::array<double, kMyCobotDof> joint_max = {
    2.879793, 1.570796, 2.879793, 2.879793, 2.879793, 2.879793};
  double workspace_x_min = -0.28;
  double workspace_x_max = 0.28;
  double workspace_y_min = -0.28;
  double workspace_y_max = 0.28;
  double workspace_z_min = 0.05;
  double workspace_z_max = 0.45;
  double joint_limit_penalty_weight = 10.0;
  double velocity_penalty_weight = 5.0;
  double workspace_penalty_weight = 8.0;
};

struct SafetyPenaltyResult
{
  double joint_limit_penalty = 0.0;
  double velocity_penalty = 0.0;
  double workspace_penalty = 0.0;
  double total_penalty = 0.0;
  bool boundary_violated = false;
};

SafetyPenaltyResult evaluate_safety_boundaries(
  const SafetyBoundaryConfig & config,
  const std::array<double, kMyCobotDof> & joint_positions,
  const std::array<double, kMyCobotDof> & joint_velocities,
  double end_effector_x,
  double end_effector_y,
  double end_effector_z);

}  // namespace spark_verify_pkg

#endif  // SPARK_VERIFY_PKG__SAFETY_BOUNDARY_EVALUATOR_HPP_
