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

#include <array>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "spark_verify_pkg/msg/policy_inference.hpp"
#include "spark_verify_pkg/msg/serial_command.hpp"
#include "spark_verify_pkg/pymycobot_serial_encoder.hpp"
#include "spark_verify_pkg/safety_boundary_evaluator.hpp"

namespace spark_verify_pkg
{

class PymycobotDriver : public rclcpp::Node
{
public:
  PymycobotDriver()
  : Node("pymycobot_driver")
  {
    declare_parameter<std::string>("inference_topic", "/mycobot/policy/inference");
    declare_parameter<std::string>("serial_topic", "/mycobot/hardware/serial_command");

    const auto inference_topic = get_parameter("inference_topic").as_string();
    const auto serial_topic = get_parameter("serial_topic").as_string();

    serial_pub_ = create_publisher<spark_verify_pkg::msg::SerialCommand>(serial_topic, 10);
    inference_sub_ = create_subscription<spark_verify_pkg::msg::PolicyInference>(
      inference_topic, 10,
      std::bind(&PymycobotDriver::on_inference, this, std::placeholders::_1));
  }

private:
  void on_inference(const spark_verify_pkg::msg::PolicyInference::SharedPtr msg)
  {
    if (!msg->accepted) {
      RCLCPP_WARN(get_logger(), "Skipping rejected policy inference message");
      return;
    }

    std::array<double, kMyCobotDof> joint_positions{};
    std::array<double, kMyCobotDof> joint_velocities{};
    for (std::size_t idx = 0; idx < kMyCobotDof; ++idx) {
      joint_positions[idx] = msg->joint_targets_rad[idx];
    }

    SafetyBoundaryConfig config;
    const auto safety = evaluate_safety_boundaries(
      config, joint_positions, joint_velocities, 0.1, 0.0, 0.2);
    if (safety.boundary_violated) {
      RCLCPP_WARN(get_logger(), "Rejected out-of-bound joint inference before serial transmit");
      return;
    }

    const auto packet = encode_send_angles_packet(joint_positions);
    spark_verify_pkg::msg::SerialCommand serial;
    serial.header = msg->header;
    serial.command_id = kPymycobotSendAnglesCommand;
    serial.payload.assign(packet.begin(), packet.end());
    serial_pub_->publish(serial);
  }

  rclcpp::Publisher<spark_verify_pkg::msg::SerialCommand>::SharedPtr serial_pub_;
  rclcpp::Subscription<spark_verify_pkg::msg::PolicyInference>::SharedPtr inference_sub_;
};

}  // namespace spark_verify_pkg

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<spark_verify_pkg::PymycobotDriver>());
  rclcpp::shutdown();
  return 0;
}
