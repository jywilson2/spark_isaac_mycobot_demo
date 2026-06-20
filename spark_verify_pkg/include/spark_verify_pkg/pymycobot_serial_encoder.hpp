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

#ifndef SPARK_VERIFY_PKG__PYMYCOBOT_SERIAL_ENCODER_HPP_
#define SPARK_VERIFY_PKG__PYMYCOBOT_SERIAL_ENCODER_HPP_

#include <array>
#include <cstdint>
#include <vector>

#include "spark_verify_pkg/articulation_model.hpp"

namespace spark_verify_pkg
{

constexpr std::uint8_t kPymycobotHeaderByte = 0xFE;
constexpr std::uint8_t kPymycobotSendAnglesCommand = 0x22;

std::vector<std::uint8_t> encode_send_angles_packet(
  const std::array<double, kMyCobotDof> & joint_angles_rad);

bool decode_send_angles_packet(
  const std::vector<std::uint8_t> & payload,
  std::array<double, kMyCobotDof> & joint_angles_rad);

}  // namespace spark_verify_pkg

#endif  // SPARK_VERIFY_PKG__PYMYCOBOT_SERIAL_ENCODER_HPP_
