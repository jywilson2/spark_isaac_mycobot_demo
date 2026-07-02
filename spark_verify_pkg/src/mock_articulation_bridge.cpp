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
// ROS 2 FUNDAMENTALS — THE NODE: A "node" is the basic unit of computation
// in a ROS graph. This file implements one in C++17 with rclcpp (the C++
// client library; rclpy is the Python equivalent). It stands in for the
// Isaac Sim articulation bridge in Phase 1:
//
//   subscribe  /mycobot/joint_commands  (sensor_msgs/JointState)
//   publish    /mycobot/joint_states    (sensor_msgs/JointState)
//   broadcast  /tf                      (per-link transforms)
//
// In the live system, Isaac Sim's ROS 2 bridge performs this same role: it
// applies JointState commands to the GPU-simulated articulation and reports
// simulated state back. Because the topic contract is identical, the tests
// written against this mock remain valid against the real simulator.
// ---------------------------------------------------------------------------

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

// Nodes are classes that inherit rclcpp::Node. Everything the node owns —
// publishers, subscriptions, timers, parameters — is created through the
// inherited API so the executor can manage callbacks and lifetimes.
class MockArticulationBridge : public rclcpp::Node
{
public:
  MockArticulationBridge()
  : Node("mock_articulation_bridge")  // node name as it appears in `ros2 node list`
  {
    // PARAMETERS: declare_parameter registers a named, typed value with a
    // default. Launch files (see phase1_mock_ecosystem.launch.py) override
    // these at startup, and `ros2 param get/set` can inspect them at
    // runtime. Making topic names parameters keeps the node remappable
    // without recompiling.
    declare_parameter<std::string>("joint_commands_topic", "/mycobot/joint_commands");
    declare_parameter<std::string>("joint_states_topic", "/mycobot/joint_states");
    declare_parameter<std::string>("base_frame", "base_link");

    const auto joint_commands_topic = get_parameter("joint_commands_topic").as_string();
    const auto joint_states_topic = get_parameter("joint_states_topic").as_string();
    base_frame_ = get_parameter("base_frame").as_string();

    // PUBLISHER: create_publisher<MsgT>(topic, qos). The trailing "10" is
    // shorthand for a QoS profile with history KEEP_LAST and depth 10 —
    // i.e. buffer up to 10 unsent messages. QoS (Quality of Service) is a
    // DDS concept; publishers and subscriptions only connect when their
    // QoS policies are compatible.
    joint_state_pub_ = create_publisher<sensor_msgs::msg::JointState>(joint_states_topic, 10);

    // TF BROADCASTER: a thin convenience wrapper around a publisher on the
    // special /tf topic. Downstream nodes use a tf2_ros::Buffer +
    // TransformListener to accumulate these into a queryable tree.
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    // SUBSCRIPTION: create_subscription<MsgT>(topic, qos, callback). The
    // callback runs when the executor (rclcpp::spin in main) dispatches an
    // incoming message — ROS nodes are event-driven, not polling loops.
    // std::bind adapts the member function into the callable the API
    // expects; a lambda would work equally well.
    joint_command_sub_ = create_subscription<sensor_msgs::msg::JointState>(
      joint_commands_topic, 10,
      std::bind(&MockArticulationBridge::on_joint_command, this, std::placeholders::_1));
  }

private:
  void on_joint_command(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    // Messages arrive as shared_ptr so multiple intraprocess subscribers
    // can share one buffer without copying (a zero-copy optimization).
    try {
      model_.set_joint_positions(msg->name, msg->position);
    } catch (const std::exception & ex) {
      // RCLCPP_ERROR routes through the ROS logging system (rosout), so
      // the message is visible in the launch console and in log files
      // under ~/.ros/log — far better than std::cerr for debugging a
      // multi-process graph.
      RCLCPP_ERROR(get_logger(), "Rejected joint command: %s", ex.what());
      return;
    }

    publish_state();
  }

  void publish_state()
  {
    // now() returns rclcpp::Time from the node's clock. Using one stamp for
    // the JointState and every TF edge keeps the whole state snapshot
    // consistent in time, which matters for tf2's interpolation.
    const auto stamp = now();
    sensor_msgs::msg::JointState state;
    state.header.stamp = stamp;
    const auto names = ArticulationModel::default_joint_names();
    state.name.assign(names.begin(), names.end());
    state.position.assign(model_.joint_positions().begin(), model_.joint_positions().end());
    joint_state_pub_->publish(state);

    // Recompute forward kinematics and broadcast one transform per link.
    // This mirrors what robot_state_publisher does for a real URDF.
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
  // Publishers/subscriptions are held as shared_ptr members; letting them
  // go out of scope would destroy the underlying DDS entities.
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_command_sub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

}  // namespace spark_verify_pkg

// Standard rclcpp entry point: init() parses ROS arguments (e.g. remaps
// passed by the launch system), spin() runs a single-threaded executor that
// blocks and services callbacks until shutdown (Ctrl-C / SIGINT), and
// shutdown() tears down the ROS context cleanly.
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<spark_verify_pkg::MockArticulationBridge>());
  rclcpp::shutdown();
  return 0;
}
