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

"""Adapt live Isaac Sim RGB frames into NITROS-style zero-copy frame handles."""

from __future__ import annotations

import os
import uuid

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from spark_verify_pkg.msg import NitrosFrameHandle


class LiveNitrosCameraBridge(Node):
    """Publish NitrosFrameHandle descriptors for live Isaac Sim camera RGB frames."""

    def __init__(self) -> None:
        super().__init__('live_nitros_camera_bridge')
        self.declare_parameter('rgb_topic', '/mycobot/camera/rgb')
        self.declare_parameter('nitros_topic', '/mycobot/camera/nitros/rgb')
        self.declare_parameter('process_id', os.getpid())

        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        nitros_topic = self.get_parameter('nitros_topic').get_parameter_value().string_value
        self._process_id = self.get_parameter('process_id').get_parameter_value().integer_value
        self._uid = str(uuid.uuid4())

        self._nitros_pub = self.create_publisher(NitrosFrameHandle, nitros_topic, 10)
        self.create_subscription(Image, rgb_topic, self._on_rgb_frame, 10)

    def _on_rgb_frame(self, image: Image) -> None:
        payload_size = int(image.step) * int(image.height)
        nitros = NitrosFrameHandle()
        nitros.header = image.header
        nitros.height = image.height
        nitros.width = image.width
        nitros.encoding = image.encoding
        nitros.step = image.step
        nitros.data = [self._process_id, 0, payload_size]
        nitros.uid = self._uid
        nitros.device_id = 0
        self._nitros_pub.publish(nitros)


def main() -> None:
    rclpy.init()
    node = LiveNitrosCameraBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
