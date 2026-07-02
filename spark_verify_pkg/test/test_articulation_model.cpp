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
// TESTING FUNDAMENTALS — UNIT vs INTEGRATION: These are GoogleTest (gtest)
// unit tests, registered with CMake via ament_add_gtest. They exercise the
// ArticulationModel LIBRARY directly — no ROS init, no processes, no
// network — so they run in milliseconds and never flake. The complementary
// launch_testing suite (test_phase1_integration.py) verifies the same math
// end-to-end through actual topics. gtest macro cheat sheet: TEST(suite,
// name) defines a case; EXPECT_* records a failure and continues; ASSERT_*
// aborts the case immediately (use it when continuing would crash, e.g.
// before indexing into a container whose size was just checked).
// ---------------------------------------------------------------------------

#include <cmath>

#include "gtest/gtest.h"
#include "spark_verify_pkg/articulation_model.hpp"

TEST(ArticulationModelTest, DefaultJointNamesMatchMyCobot)
{
  // Guards the naming contract shared with the URDF and the Python nodes:
  // if someone renames joints, this fails before any runtime mismatch.
  const auto names = spark_verify_pkg::ArticulationModel::default_joint_names();
  ASSERT_EQ(names.size(), spark_verify_pkg::kMyCobotDof);
  EXPECT_EQ(names[0], "joint1");
  EXPECT_EQ(names[5], "joint6");
}

TEST(ArticulationModelTest, JointCommandsUpdateLinkTransforms)
{
  spark_verify_pkg::ArticulationModel model;
  // Deliberately command a SUBSET of joints (only joint1/joint2) to prove
  // the model matches by name rather than requiring all six.
  std::vector<std::string> names = {"joint1", "joint2"};
  std::vector<double> positions = {0.5, -0.25};
  model.set_joint_positions(names, positions);

  const auto transforms = model.compute_link_transforms("base_link", "base_link", 0.0, 0U);
  ASSERT_EQ(transforms.size(), spark_verify_pkg::kMyCobotDof);

  // Independently recompute the expected forward kinematics for link1
  // (offset 0.1315 m rotated by 0.5 rad) rather than trusting the model's
  // own math — a test that re-runs the code under test proves nothing.
  const double expected_x = 0.1315 * std::cos(0.5);
  const double expected_y = 0.1315 * std::sin(0.5);
  // EXPECT_NEAR for floating point: exact equality comparisons on doubles
  // fail on harmless last-bit differences.
  EXPECT_NEAR(transforms[0].transform.translation.x, expected_x, 1e-6);
  EXPECT_NEAR(transforms[0].transform.translation.y, expected_y, 1e-6);
}

TEST(ArticulationModelTest, RejectsUnknownJointName)
{
  // Error-path coverage: a typo'd joint name must throw, not be silently
  // dropped. EXPECT_THROW asserts both that it throws and the exact type.
  spark_verify_pkg::ArticulationModel model;
  EXPECT_THROW(
    model.set_joint_positions({"unknown_joint"}, {0.1}),
    std::invalid_argument);
}
