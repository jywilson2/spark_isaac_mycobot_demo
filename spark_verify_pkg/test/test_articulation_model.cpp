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

#include <cmath>

#include "gtest/gtest.h"
#include "spark_verify_pkg/articulation_model.hpp"

TEST(ArticulationModelTest, DefaultJointNamesMatchMyCobot)
{
  const auto names = spark_verify_pkg::ArticulationModel::default_joint_names();
  ASSERT_EQ(names.size(), spark_verify_pkg::kMyCobotDof);
  EXPECT_EQ(names[0], "joint1");
  EXPECT_EQ(names[5], "joint6");
}

TEST(ArticulationModelTest, JointCommandsUpdateLinkTransforms)
{
  spark_verify_pkg::ArticulationModel model;
  std::vector<std::string> names = {"joint1", "joint2"};
  std::vector<double> positions = {0.5, -0.25};
  model.set_joint_positions(names, positions);

  const auto transforms = model.compute_link_transforms("base_link", "base_link", 0.0, 0U);
  ASSERT_EQ(transforms.size(), spark_verify_pkg::kMyCobotDof);

  const double expected_x = 0.1315 * std::cos(0.5);
  const double expected_y = 0.1315 * std::sin(0.5);
  EXPECT_NEAR(transforms[0].transform.translation.x, expected_x, 1e-6);
  EXPECT_NEAR(transforms[0].transform.translation.y, expected_y, 1e-6);
}

TEST(ArticulationModelTest, RejectsUnknownJointName)
{
  spark_verify_pkg::ArticulationModel model;
  EXPECT_THROW(
    model.set_joint_positions({"unknown_joint"}, {0.1}),
    std::invalid_argument);
}
