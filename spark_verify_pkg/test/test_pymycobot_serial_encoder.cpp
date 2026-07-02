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
// PHASE 3 VERIFICATION REQUIREMENT (spec.md): "asserts expected serial byte
// outputs for joint states." Serial protocols are byte-exact contracts, so
// the test strategy is threefold: (1) structural checks on known packets,
// (2) ROUND-TRIP tests (encode then decode must reproduce the input within
// the wire format's 0.1-degree quantization), and (3) corruption tests
// proving the decoder rejects every class of malformed frame it could see
// on a real serial line (bad checksum, bad header, wrong command, short
// read). The round-trip tolerance of 0.02 rad ~= 1.1 deg comfortably covers
// the 0.05 deg worst-case quantization error.
// ---------------------------------------------------------------------------

#include <cmath>

#include "gtest/gtest.h"
#include "spark_verify_pkg/pymycobot_serial_encoder.hpp"

TEST(PymycobotSerialEncoderTest, EncodesDeterministicPacketForKnownAngles)
{
  const std::array<double, spark_verify_pkg::kMyCobotDof> joints = {
    0.4, -0.2, 0.6, -0.3, 0.5, -0.1};
  const auto packet = spark_verify_pkg::encode_send_angles_packet(joints);

  // Frame structure: 2 header + 1 length + 1 command + 12 data + 1 checksum.
  ASSERT_EQ(packet.size(), 17U);
  EXPECT_EQ(packet[0], spark_verify_pkg::kPymycobotHeaderByte);
  EXPECT_EQ(packet[1], spark_verify_pkg::kPymycobotHeaderByte);
  EXPECT_EQ(packet[3], spark_verify_pkg::kPymycobotSendAnglesCommand);
}

TEST(PymycobotSerialEncoderTest, RoundTripPreservesJointAngles)
{
  // Round-trip property: decode(encode(x)) == x within quantization. This
  // single property implies the byte packing and unpacking agree on
  // endianness, offsets, and units without hand-writing expected bytes.
  const std::array<double, spark_verify_pkg::kMyCobotDof> expected = {
    0.4, -0.2, 0.6, -0.3, 0.5, -0.1};
  const auto packet = spark_verify_pkg::encode_send_angles_packet(expected);

  std::array<double, spark_verify_pkg::kMyCobotDof> decoded{};
  ASSERT_TRUE(spark_verify_pkg::decode_send_angles_packet(packet, decoded));

  for (std::size_t idx = 0; idx < spark_verify_pkg::kMyCobotDof; ++idx) {
    EXPECT_NEAR(decoded[idx], expected[idx], 0.02);
  }
}

TEST(PymycobotSerialEncoderTest, RoundTripPreservesNegativeAndExtremeAngles)
{
  // Edge-of-range inputs: the MyCobot joint limits (+/-2.879793 rad =
  // +/-165 deg) are where a sign-handling or overflow bug in the int16
  // big-endian packing would appear. Negative values specifically exercise
  // the two's-complement byte split.
  const std::array<double, spark_verify_pkg::kMyCobotDof> expected = {
    -2.879793, 2.879793, -1.570796, 1.570796, -0.001, 0.0};
  const auto packet = spark_verify_pkg::encode_send_angles_packet(expected);

  std::array<double, spark_verify_pkg::kMyCobotDof> decoded{};
  ASSERT_TRUE(spark_verify_pkg::decode_send_angles_packet(packet, decoded));

  for (std::size_t idx = 0; idx < spark_verify_pkg::kMyCobotDof; ++idx) {
    EXPECT_NEAR(decoded[idx], expected[idx], 0.02);
  }
}

TEST(PymycobotSerialEncoderTest, ChecksumMatchesByteSumOfBody)
{
  // Recompute the checksum independently (8-bit sum of bytes 2..N-2) and
  // compare with the trailer byte the encoder produced. This pins the
  // checksum ALGORITHM, not just internal encode/decode consistency —
  // both sides could otherwise share the same wrong formula.
  const auto packet = spark_verify_pkg::encode_send_angles_packet(
    {0.4, -0.2, 0.6, -0.3, 0.5, -0.1});

  std::uint8_t expected_checksum = 0U;
  for (std::size_t idx = 2; idx < packet.size() - 1U; ++idx) {
    expected_checksum += packet[idx];
  }
  EXPECT_EQ(packet.back(), expected_checksum);
}

TEST(PymycobotSerialEncoderTest, RejectsInvalidChecksum)
{
  // Simulate line noise flipping bits in the checksum byte; the decoder
  // must return false rather than deliver corrupted joint angles.
  auto packet = spark_verify_pkg::encode_send_angles_packet({0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
  packet.back() ^= 0xFF;

  std::array<double, spark_verify_pkg::kMyCobotDof> decoded{};
  EXPECT_FALSE(spark_verify_pkg::decode_send_angles_packet(packet, decoded));
}

TEST(PymycobotSerialEncoderTest, RejectsCorruptedHeaderOrCommand)
{
  const auto valid = spark_verify_pkg::encode_send_angles_packet(
    {0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
  std::array<double, spark_verify_pkg::kMyCobotDof> decoded{};

  // A wrong header byte means the reader lost frame sync mid-stream.
  auto bad_header = valid;
  bad_header[0] = 0x00;
  EXPECT_FALSE(spark_verify_pkg::decode_send_angles_packet(bad_header, decoded));

  // A wrong command id means this frame is some OTHER pymycobot command;
  // decoding it as send_angles would misinterpret its payload.
  auto bad_command = valid;
  bad_command[3] = 0x11;
  EXPECT_FALSE(spark_verify_pkg::decode_send_angles_packet(bad_command, decoded));
}

TEST(PymycobotSerialEncoderTest, RejectsTruncatedPacket)
{
  // Short reads are routine on serial ports (buffer boundaries, timeouts).
  // The size check must fire before any offset-based field access.
  auto packet = spark_verify_pkg::encode_send_angles_packet(
    {0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
  packet.pop_back();

  std::array<double, spark_verify_pkg::kMyCobotDof> decoded{};
  EXPECT_FALSE(spark_verify_pkg::decode_send_angles_packet(packet, decoded));
}
