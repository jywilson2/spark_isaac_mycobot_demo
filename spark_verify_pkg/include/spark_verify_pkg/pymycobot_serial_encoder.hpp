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
// PROJECT CONTEXT: The physical MyCobot 280 is commanded over a USB serial
// link. The vendor's Python library, pymycobot, frames every command as:
//
//   [0xFE, 0xFE, length, command, ...payload..., footer/checksum]
//
// with two 0xFE header bytes and per-joint angles packed as big-endian
// signed 16-bit integers in units of 0.1 degree ("centidegree * 10"). This
// file declares a deterministic mock of that framing so Phase 3 tests can
// assert exact byte output BEFORE any hardware is attached — the essence of
// sim-to-real preparation: freeze the wire contract, then swap in hardware.
// ---------------------------------------------------------------------------

#ifndef SPARK_VERIFY_PKG__PYMYCOBOT_SERIAL_ENCODER_HPP_
#define SPARK_VERIFY_PKG__PYMYCOBOT_SERIAL_ENCODER_HPP_

#include <array>
#include <cstdint>
#include <vector>

#include "spark_verify_pkg/articulation_model.hpp"  // for kMyCobotDof

namespace spark_verify_pkg
{

// Frame header byte, repeated twice at the start of every packet so the
// receiver can resynchronize mid-stream after dropped bytes.
constexpr std::uint8_t kPymycobotHeaderByte = 0xFE;
// 0x22 is pymycobot's "send_angles" command id (write all joint angles).
constexpr std::uint8_t kPymycobotSendAnglesCommand = 0x22;

// Encode six joint angles (radians) into a 17-byte send_angles packet:
// 2 header + 1 length + 1 command + 6*2 angle bytes + 1 checksum.
std::vector<std::uint8_t> encode_send_angles_packet(
  const std::array<double, kMyCobotDof> & joint_angles_rad);

// Decode and validate a packet produced by encode_send_angles_packet.
// Returns false (instead of throwing) on any framing problem — wrong size,
// bad header, wrong command, checksum mismatch — because on a real serial
// line corrupted packets are an expected condition, not an exception.
bool decode_send_angles_packet(
  const std::vector<std::uint8_t> & payload,
  std::array<double, kMyCobotDof> & joint_angles_rad);

}  // namespace spark_verify_pkg

#endif  // SPARK_VERIFY_PKG__PYMYCOBOT_SERIAL_ENCODER_HPP_
