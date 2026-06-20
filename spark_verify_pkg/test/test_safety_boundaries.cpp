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

#include "gtest/gtest.h"
#include "spark_verify_pkg/safety_boundary_evaluator.hpp"

namespace
{

spark_verify_pkg::SafetyBoundaryConfig default_config()
{
  return spark_verify_pkg::SafetyBoundaryConfig{};
}

std::array<double, spark_verify_pkg::kMyCobotDof> zero_state()
{
  return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

}  // namespace

TEST(SafetyBoundaryEvaluatorTest, SafeStateProducesZeroPenalty)
{
  const auto config = default_config();
  const auto joints = zero_state();
  const auto velocities = zero_state();

  const auto result = spark_verify_pkg::evaluate_safety_boundaries(
    config, joints, velocities, 0.1, 0.0, 0.2);

  EXPECT_FALSE(result.boundary_violated);
  EXPECT_DOUBLE_EQ(result.total_penalty, 0.0);
}

TEST(SafetyBoundaryEvaluatorTest, JointLimitViolationTriggersPenalty)
{
  const auto config = default_config();
  auto joints = zero_state();
  joints[1] = config.joint_max[1] + 0.2;
  const auto velocities = zero_state();

  const auto result = spark_verify_pkg::evaluate_safety_boundaries(
    config, joints, velocities, 0.1, 0.0, 0.2);

  EXPECT_TRUE(result.boundary_violated);
  EXPECT_GT(result.joint_limit_penalty, 0.0);
  EXPECT_DOUBLE_EQ(result.velocity_penalty, 0.0);
  EXPECT_DOUBLE_EQ(result.workspace_penalty, 0.0);
  EXPECT_NEAR(result.total_penalty, 2.0, 1e-6);
}

TEST(SafetyBoundaryEvaluatorTest, VelocityViolationTriggersPenalty)
{
  const auto config = default_config();
  const auto joints = zero_state();
  auto velocities = zero_state();
  velocities[0] = config.max_joint_velocity_rad_s + 0.5;

  const auto result = spark_verify_pkg::evaluate_safety_boundaries(
    config, joints, velocities, 0.1, 0.0, 0.2);

  EXPECT_TRUE(result.boundary_violated);
  EXPECT_DOUBLE_EQ(result.joint_limit_penalty, 0.0);
  EXPECT_NEAR(result.velocity_penalty, 2.5, 1e-6);
  EXPECT_DOUBLE_EQ(result.workspace_penalty, 0.0);
}

TEST(SafetyBoundaryEvaluatorTest, WorkspaceViolationTriggersPenalty)
{
  const auto config = default_config();
  const auto joints = zero_state();
  const auto velocities = zero_state();

  const auto result = spark_verify_pkg::evaluate_safety_boundaries(
    config, joints, velocities, 0.0, 0.0, config.workspace_z_max + 0.1);

  EXPECT_TRUE(result.boundary_violated);
  EXPECT_DOUBLE_EQ(result.joint_limit_penalty, 0.0);
  EXPECT_DOUBLE_EQ(result.velocity_penalty, 0.0);
  EXPECT_NEAR(result.workspace_penalty, 0.8, 1e-6);
}
