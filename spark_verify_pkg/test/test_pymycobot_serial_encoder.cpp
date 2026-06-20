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
#include "spark_verify_pkg/pymycobot_serial_encoder.hpp"

TEST(PymycobotSerialEncoderTest, EncodesDeterministicPacketForKnownAngles)
{
  const std::array<double, spark_verify_pkg::kMyCobotDof> joints = {
    0.4, -0.2, 0.6, -0.3, 0.5, -0.1};
  const auto packet = spark_verify_pkg::encode_send_angles_packet(joints);

  ASSERT_EQ(packet.size(), 17U);
  EXPECT_EQ(packet[0], spark_verify_pkg::kPymycobotHeaderByte);
  EXPECT_EQ(packet[1], spark_verify_pkg::kPymycobotHeaderByte);
  EXPECT_EQ(packet[3], spark_verify_pkg::kPymycobotSendAnglesCommand);
}

TEST(PymycobotSerialEncoderTest, RoundTripPreservesJointAngles)
{
  const std::array<double, spark_verify_pkg::kMyCobotDof> expected = {
    0.4, -0.2, 0.6, -0.3, 0.5, -0.1};
  const auto packet = spark_verify_pkg::encode_send_angles_packet(expected);

  std::array<double, spark_verify_pkg::kMyCobotDof> decoded{};
  ASSERT_TRUE(spark_verify_pkg::decode_send_angles_packet(packet, decoded));

  for (std::size_t idx = 0; idx < spark_verify_pkg::kMyCobotDof; ++idx) {
    EXPECT_NEAR(decoded[idx], expected[idx], 0.02);
  }
}

TEST(PymycobotSerialEncoderTest, RejectsInvalidChecksum)
{
  auto packet = spark_verify_pkg::encode_send_angles_packet({0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
  packet.back() ^= 0xFF;

  std::array<double, spark_verify_pkg::kMyCobotDof> decoded{};
  EXPECT_FALSE(spark_verify_pkg::decode_send_angles_packet(packet, decoded));
}
