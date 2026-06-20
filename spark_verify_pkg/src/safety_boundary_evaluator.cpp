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

#include "spark_verify_pkg/safety_boundary_evaluator.hpp"

namespace spark_verify_pkg
{

namespace
{

double compute_joint_limit_penalty(
  const SafetyBoundaryConfig & config,
  const std::array<double, kMyCobotDof> & joint_positions)
{
  double penalty = 0.0;
  for (std::size_t idx = 0; idx < kMyCobotDof; ++idx) {
    if (joint_positions[idx] < config.joint_min[idx]) {
      penalty += config.joint_limit_penalty_weight *
        std::abs(joint_positions[idx] - config.joint_min[idx]);
    } else if (joint_positions[idx] > config.joint_max[idx]) {
      penalty += config.joint_limit_penalty_weight *
        std::abs(joint_positions[idx] - config.joint_max[idx]);
    }
  }
  return penalty;
}

double compute_velocity_penalty(
  const SafetyBoundaryConfig & config,
  const std::array<double, kMyCobotDof> & joint_velocities)
{
  double penalty = 0.0;
  for (const double velocity : joint_velocities) {
    if (std::abs(velocity) > config.max_joint_velocity_rad_s) {
      penalty += config.velocity_penalty_weight *
        (std::abs(velocity) - config.max_joint_velocity_rad_s);
    }
  }
  return penalty;
}

double compute_workspace_penalty(
  const SafetyBoundaryConfig & config,
  double end_effector_x,
  double end_effector_y,
  double end_effector_z)
{
  double penalty = 0.0;
  if (end_effector_x < config.workspace_x_min) {
    penalty += config.workspace_penalty_weight *
      (config.workspace_x_min - end_effector_x);
  } else if (end_effector_x > config.workspace_x_max) {
    penalty += config.workspace_penalty_weight *
      (end_effector_x - config.workspace_x_max);
  }

  if (end_effector_y < config.workspace_y_min) {
    penalty += config.workspace_penalty_weight *
      (config.workspace_y_min - end_effector_y);
  } else if (end_effector_y > config.workspace_y_max) {
    penalty += config.workspace_penalty_weight *
      (end_effector_y - config.workspace_y_max);
  }

  if (end_effector_z < config.workspace_z_min) {
    penalty += config.workspace_penalty_weight *
      (config.workspace_z_min - end_effector_z);
  } else if (end_effector_z > config.workspace_z_max) {
    penalty += config.workspace_penalty_weight *
      (end_effector_z - config.workspace_z_max);
  }

  return penalty;
}

}  // namespace

SafetyPenaltyResult evaluate_safety_boundaries(
  const SafetyBoundaryConfig & config,
  const std::array<double, kMyCobotDof> & joint_positions,
  const std::array<double, kMyCobotDof> & joint_velocities,
  double end_effector_x,
  double end_effector_y,
  double end_effector_z)
{
  SafetyPenaltyResult result;
  result.joint_limit_penalty = compute_joint_limit_penalty(config, joint_positions);
  result.velocity_penalty = compute_velocity_penalty(config, joint_velocities);
  result.workspace_penalty = compute_workspace_penalty(
    config, end_effector_x, end_effector_y, end_effector_z);
  result.total_penalty =
    result.joint_limit_penalty + result.velocity_penalty + result.workspace_penalty;
  result.boundary_violated = result.total_penalty > 0.0;
  return result;
}

}  // namespace spark_verify_pkg
