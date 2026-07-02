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
// PROJECT CONTEXT: In reinforcement learning for robots, "safety boundaries"
// serve two purposes. During TRAINING (Phase 2, Isaac Lab) they become
// penalty terms in the reward so the policy learns to avoid joint limits,
// runaway velocities, and leaving the workspace. During DEPLOYMENT (Phases
// 3-4) the same evaluator becomes a hard gate: the pymycobot driver refuses
// to transmit any serial command that violates a boundary, protecting the
// physical arm from an unstable or adversarial policy output. Sharing one
// deterministic implementation for both uses is what makes the behaviour
// testable before any training run.
// ---------------------------------------------------------------------------

#ifndef SPARK_VERIFY_PKG__SAFETY_BOUNDARY_EVALUATOR_HPP_
#define SPARK_VERIFY_PKG__SAFETY_BOUNDARY_EVALUATOR_HPP_

#include <array>
#include <cmath>

#include "spark_verify_pkg/articulation_model.hpp"

namespace spark_verify_pkg
{

// All limits live in one aggregate so a node could later expose them as ROS
// parameters and pass a customized config into the evaluator.
struct SafetyBoundaryConfig
{
  // MyCobot 280 joints top out at 160 deg/s (~2.8 rad/s); 2.0 rad/s leaves
  // a safety margin below the hardware maximum.
  double max_joint_velocity_rad_s = 2.0;
  // Joint limits in radians. The MyCobot 280 spec sheet lists +/-165 deg
  // (2.879793 rad) for most joints; joint2 is restricted further here to
  // keep the mock workspace conservative.
  std::array<double, kMyCobotDof> joint_min = {
    -2.879793, -1.570796, -2.879793, -2.879793, -2.879793, -2.879793};
  std::array<double, kMyCobotDof> joint_max = {
    2.879793, 1.570796, 2.879793, 2.879793, 2.879793, 2.879793};
  // Axis-aligned bounding box (meters, in the base_link frame) that the end
  // effector must stay inside. The MyCobot 280 working radius is 0.28 m,
  // hence the +/-0.28 m X/Y extents.
  double workspace_x_min = -0.28;
  double workspace_x_max = 0.28;
  double workspace_y_min = -0.28;
  double workspace_y_max = 0.28;
  double workspace_z_min = 0.05;
  double workspace_z_max = 0.45;
  // Penalty weights convert a violation magnitude (radians or meters past
  // the limit) into a scalar cost. In RL these scale how strongly the
  // reward discourages each violation class.
  double joint_limit_penalty_weight = 10.0;
  double velocity_penalty_weight = 5.0;
  double workspace_penalty_weight = 8.0;
};

// The evaluator returns each penalty term separately (useful for reward
// shaping diagnostics and for the gtest assertions) plus the aggregate and
// a boolean that deployment code uses as a hard go/no-go gate.
struct SafetyPenaltyResult
{
  double joint_limit_penalty = 0.0;
  double velocity_penalty = 0.0;
  double workspace_penalty = 0.0;
  double total_penalty = 0.0;
  bool boundary_violated = false;
};

// Pure function: no ROS types, no state, deterministic output for a given
// input. That makes it trivially unit-testable (test_safety_boundaries.cpp)
// and safe to call from both the RL reward path and the driver hot path.
SafetyPenaltyResult evaluate_safety_boundaries(
  const SafetyBoundaryConfig & config,
  const std::array<double, kMyCobotDof> & joint_positions,
  const std::array<double, kMyCobotDof> & joint_velocities,
  double end_effector_x,
  double end_effector_y,
  double end_effector_z);

}  // namespace spark_verify_pkg

#endif  // SPARK_VERIFY_PKG__SAFETY_BOUNDARY_EVALUATOR_HPP_
