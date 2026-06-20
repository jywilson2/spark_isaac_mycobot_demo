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

"""Mock workspace camera publisher with NITROS-style zero-copy frame handles."""

import os
import uuid

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from spark_verify_pkg.msg import NitrosFrameHandle


class MockCameraPublisher(Node):
    """Publishes RGB frames and NITROS zero-copy descriptor handles."""

    def __init__(self) -> None:
        super().__init__('mock_camera_publisher')
        self.declare_parameter('rgb_topic', '/mycobot/camera/rgb')
        self.declare_parameter('nitros_topic', '/mycobot/camera/nitros/rgb')
        self.declare_parameter('frame_id', 'camera_optical_frame')
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('block_center_x', 0.5)
        self.declare_parameter('block_center_y', 0.5)
        self.declare_parameter('block_width', 0.125)
        self.declare_parameter('block_height', 0.167)

        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        nitros_topic = self.get_parameter('nitros_topic').get_parameter_value().string_value
        self._frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self._width = self.get_parameter('image_width').get_parameter_value().integer_value
        self._height = self.get_parameter('image_height').get_parameter_value().integer_value
        self._block_center_x = (
            self.get_parameter('block_center_x').get_parameter_value().double_value)
        self._block_center_y = (
            self.get_parameter('block_center_y').get_parameter_value().double_value)
        self._block_width = self.get_parameter('block_width').get_parameter_value().double_value
        self._block_height = (
            self.get_parameter('block_height').get_parameter_value().double_value)
        rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        self._rgb_pub = self.create_publisher(Image, rgb_topic, 10)
        self._nitros_pub = self.create_publisher(NitrosFrameHandle, nitros_topic, 10)
        self._mock_pid = os.getpid()
        self._mock_fd = 42
        self._uid = str(uuid.uuid4())
        self._timer = self.create_timer(1.0 / max(rate_hz, 1.0), self._publish_frames)

    def _publish_frames(self) -> None:
        stamp = self.get_clock().now().to_msg()
        row_step = self._width * 3
        payload_size = row_step * self._height

        rgb = Image()
        rgb.header.stamp = stamp
        rgb.header.frame_id = self._frame_id
        rgb.height = self._height
        rgb.width = self._width
        rgb.encoding = 'rgb8'
        rgb.is_bigendian = 0
        rgb.step = row_step
        rgb.data = [40, 40, 40] * (self._width * self._height)
        self._paint_block(rgb)
        self._rgb_pub.publish(rgb)

        nitros = NitrosFrameHandle()
        nitros.header = rgb.header
        nitros.height = self._height
        nitros.width = self._width
        nitros.encoding = 'rgb8'
        nitros.step = row_step
        nitros.data = [self._mock_pid, self._mock_fd, payload_size]
        nitros.uid = self._uid
        nitros.device_id = 0
        self._nitros_pub.publish(nitros)

    def _paint_block(self, rgb: Image) -> None:
        half_width = int(self._block_width * self._width / 2.0)
        half_height = int(self._block_height * self._height / 2.0)
        center_x = int(self._block_center_x * self._width)
        center_y = int(self._block_center_y * self._height)
        min_x = max(center_x - half_width, 0)
        max_x = min(center_x + half_width, self._width)
        min_y = max(center_y - half_height, 0)
        max_y = min(center_y + half_height, self._height)

        for row in range(min_y, max_y):
            row_offset = row * rgb.step
            for col in range(min_x, max_x):
                pixel_offset = row_offset + col * 3
                rgb.data[pixel_offset] = 230
                rgb.data[pixel_offset + 1] = 30
                rgb.data[pixel_offset + 2] = 30


def main() -> None:
    rclpy.init()
    node = MockCameraPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
