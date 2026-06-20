# Copyright 2026 spark_isaac_mycobot_demo contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ROS node that publishes block detections from the mock camera stream."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from spark_verify_nodes.block_vision_tracker import to_block_detection
from spark_verify_pkg.msg import BlockDetection


class BlockVisionTrackerNode(Node):
    """Track the synthetic workspace block and publish normalized detections."""

    def __init__(self) -> None:
        super().__init__('block_vision_tracker')
        self.declare_parameter('rgb_topic', '/mycobot/camera/rgb')
        self.declare_parameter('detection_topic', '/mycobot/vision/block_detection')
        self.declare_parameter('red_threshold', 180)

        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        red_threshold = self.get_parameter('red_threshold').get_parameter_value().integer_value

        self._publisher = self.create_publisher(BlockDetection, detection_topic, 10)
        self._red_threshold = red_threshold
        self._subscription = self.create_subscription(Image, rgb_topic, self._on_rgb_frame, 10)

    def _on_rgb_frame(self, image: Image) -> None:
        detection = to_block_detection(image, red_threshold=self._red_threshold)
        self._publisher.publish(detection)


def main() -> None:
    rclpy.init()
    node = BlockVisionTrackerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
