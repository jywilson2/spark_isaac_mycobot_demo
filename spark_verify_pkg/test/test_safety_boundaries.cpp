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
// PHASE 2 VERIFICATION REQUIREMENT (spec.md): "deterministic unit tests for
// safety boundaries ensuring that penalization metrics trigger correctly
// BEFORE training execution." The strategy is one test per violation class,
// each proving (a) the targeted penalty term fires with the exact expected
// magnitude, and (b) every OTHER term stays exactly zero — so the terms are
// verifiably independent and a regression in one cannot hide in another.
// ---------------------------------------------------------------------------

#include "gtest/gtest.h"
#include "spark_verify_pkg/safety_boundary_evaluator.hpp"

namespace
{

// Test fixtures/helpers: a fresh default config and an all-zero joint
// state, giving every case the same known-safe starting point.
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
  // Baseline (negative control): a nominal state inside every limit must
  // produce exactly zero penalty. Without this case, a bug that penalizes
  // everything would still pass the violation tests below.
  const auto config = default_config();
  const auto joints = zero_state();
  const auto velocities = zero_state();

  // (0.1, 0.0, 0.2) is an end-effector point comfortably inside the
  // +/-0.28 m XY and 0.05-0.45 m Z workspace box.
  const auto result = spark_verify_pkg::evaluate_safety_boundaries(
    config, joints, velocities, 0.1, 0.0, 0.2);

  EXPECT_FALSE(result.boundary_violated);
  EXPECT_DOUBLE_EQ(result.total_penalty, 0.0);
}

TEST(SafetyBoundaryEvaluatorTest, JointLimitViolationTriggersPenalty)
{
  const auto config = default_config();
  auto joints = zero_state();
  // Push joint2 exactly 0.2 rad past its upper limit. Expected penalty is
  // hand-computed from the linear hinge: weight (10.0) x overshoot (0.2)
  // = 2.0 — asserting the exact value catches weight regressions too.
  joints[1] = config.joint_max[1] + 0.2;
  const auto velocities = zero_state();

  const auto result = spark_verify_pkg::evaluate_safety_boundaries(
    config, joints, velocities, 0.1, 0.0, 0.2);

  EXPECT_TRUE(result.boundary_violated);
  EXPECT_GT(result.joint_limit_penalty, 0.0);
  // Independence: the other two penalty classes must remain untouched.
  EXPECT_DOUBLE_EQ(result.velocity_penalty, 0.0);
  EXPECT_DOUBLE_EQ(result.workspace_penalty, 0.0);
  EXPECT_NEAR(result.total_penalty, 2.0, 1e-6);
}

TEST(SafetyBoundaryEvaluatorTest, VelocityViolationTriggersPenalty)
{
  const auto config = default_config();
  const auto joints = zero_state();
  auto velocities = zero_state();
  // 0.5 rad/s over the 2.0 rad/s cap: expected 5.0 x 0.5 = 2.5.
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

  // End effector 0.1 m above the workspace ceiling: 8.0 x 0.1 = 0.8.
  const auto result = spark_verify_pkg::evaluate_safety_boundaries(
    config, joints, velocities, 0.0, 0.0, config.workspace_z_max + 0.1);

  EXPECT_TRUE(result.boundary_violated);
  EXPECT_DOUBLE_EQ(result.joint_limit_penalty, 0.0);
  EXPECT_DOUBLE_EQ(result.velocity_penalty, 0.0);
  EXPECT_NEAR(result.workspace_penalty, 0.8, 1e-6);
}
