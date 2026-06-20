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

#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "spark_verify_pkg/articulation_model.hpp"
#include "tf2_ros/transform_broadcaster.h"

namespace spark_verify_pkg
{

class MockArticulationBridge : public rclcpp::Node
{
public:
  MockArticulationBridge()
  : Node("mock_articulation_bridge")
  {
    declare_parameter<std::string>("joint_commands_topic", "/mycobot/joint_commands");
    declare_parameter<std::string>("joint_states_topic", "/mycobot/joint_states");
    declare_parameter<std::string>("base_frame", "base_link");

    const auto joint_commands_topic = get_parameter("joint_commands_topic").as_string();
    const auto joint_states_topic = get_parameter("joint_states_topic").as_string();
    base_frame_ = get_parameter("base_frame").as_string();

    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>(joint_states_topic, 10);
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    joint_command_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_commands_topic, 10,
      std::bind(&MockArticulationBridge::on_joint_command, this, std::placeholders::_1));
  }

private:
  void on_joint_command(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    try {
      model_.set_joint_positions(msg->name, msg->position);
    } catch (const std::exception & ex) {
      RCLCPP_ERROR(get_logger(), "Rejected joint command: %s", ex.what());
      return;
    }

    publish_state();
  }

  void publish_state()
  {
    const auto stamp = now();
    sensor_msgs::msg::JointState state;
    state.header.stamp = stamp;
    const auto names = ArticulationModel::default_joint_names();
    state.name.assign(names.begin(), names.end());
    state.position.assign(model_.joint_positions().begin(), model_.joint_positions().end());
    joint_state_pub_->publish(state);

    const auto transforms = model_.compute_link_transforms(
      base_frame_, base_frame_,
      stamp.seconds(), static_cast<uint32_t>(stamp.nanoseconds() % 1000000000ULL));
    for (const auto & tf : transforms) {
      geometry_msgs::msg::TransformStamped stamped = tf;
      stamped.header.stamp = stamp;
      tf_broadcaster_->sendTransform(stamped);
    }
  }

  std::string base_frame_;
  ArticulationModel model_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_command_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace spark_verify_pkg

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<spark_verify_pkg::MockArticulationBridge>());
  rclcpp::shutdown();
  return 0;
}
