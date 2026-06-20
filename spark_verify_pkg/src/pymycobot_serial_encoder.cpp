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

#include "spark_verify_pkg/pymycobot_serial_encoder.hpp"

#include <cmath>
#include <stdexcept>

namespace spark_verify_pkg
{

namespace
{

constexpr std::size_t kPacketSize = 17;

std::int16_t radians_to_protocol_value(double radians)
{
  const double degrees = radians * 180.0 / M_PI;
  return static_cast<std::int16_t>(std::lround(degrees * 10.0));
}

double protocol_value_to_radians(std::int16_t raw_value)
{
  return static_cast<double>(raw_value) / 10.0 * M_PI / 180.0;
}

}  // namespace

std::vector<std::uint8_t> encode_send_angles_packet(
  const std::array<double, kMyCobotDof> & joint_angles_rad)
{
  std::vector<std::uint8_t> packet(kPacketSize);
  packet[0] = kPymycobotHeaderByte;
  packet[1] = kPymycobotHeaderByte;
  packet[2] = static_cast<std::uint8_t>(kPacketSize - 2);
  packet[3] = kPymycobotSendAnglesCommand;

  for (std::size_t idx = 0; idx < kMyCobotDof; ++idx) {
    const std::int16_t encoded = radians_to_protocol_value(joint_angles_rad[idx]);
    const std::size_t offset = 4U + (idx * 2U);
    packet[offset] = static_cast<std::uint8_t>((encoded >> 8) & 0xFF);
    packet[offset + 1U] = static_cast<std::uint8_t>(encoded & 0xFF);
  }

  std::uint8_t checksum = 0U;
  for (std::size_t idx = 2; idx < packet.size() - 1U; ++idx) {
    checksum += packet[idx];
  }
  packet.back() = checksum;
  return packet;
}

bool decode_send_angles_packet(
  const std::vector<std::uint8_t> & payload,
  std::array<double, kMyCobotDof> & joint_angles_rad)
{
  if (payload.size() != kPacketSize) {
    return false;
  }
  if (payload[0] != kPymycobotHeaderByte || payload[1] != kPymycobotHeaderByte) {
    return false;
  }
  if (payload[3] != kPymycobotSendAnglesCommand) {
    return false;
  }

  std::uint8_t checksum = 0U;
  for (std::size_t idx = 2; idx < payload.size() - 1U; ++idx) {
    checksum += payload[idx];
  }
  if (checksum != payload.back()) {
    return false;
  }

  for (std::size_t idx = 0; idx < kMyCobotDof; ++idx) {
    const std::size_t offset = 4U + (idx * 2U);
    const std::int16_t encoded = static_cast<std::int16_t>(
      (static_cast<std::uint16_t>(payload[offset]) << 8) | payload[offset + 1U]);
    joint_angles_rad[idx] = protocol_value_to_radians(encoded);
  }
  return true;
}

}  // namespace spark_verify_pkg
