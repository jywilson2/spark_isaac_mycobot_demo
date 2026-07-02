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

"""
Mock USB camera frame source for Phase 4 HIL verification.

HIL BACKGROUND: Hardware-in-the-loop (HIL) testing exercises the deployment
software against interfaces shaped exactly like the real hardware's. On the
Raspberry Pi, a V4L2 driver (e.g. via v4l2_camera or usb_cam) would publish
sensor_msgs/Image from the USB camera; this mock publishes identically-typed
frames at a realistic rate (15 Hz) plus a cumulative frame counter that the
edge node uses to detect dropped frames. The Phase 4 test then asserts the
frame pipeline is lossless end to end.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import UInt32


class MockUsbCameraHilNode(Node):
    """Publish deterministic USB camera frames and frame-drop statistics."""

    def __init__(self) -> None:
        super().__init__('mock_usb_camera_hil')
        self.declare_parameter('frame_topic', '/edge/usb_camera/frame')
        self.declare_parameter('stats_topic', '/edge/usb_camera/stats')
        # 15 Hz approximates a budget USB camera under load; the edge
        # node's health checks must hold at this cadence.
        self.declare_parameter('publish_rate_hz', 15.0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        frame_topic = self.get_parameter('frame_topic').get_parameter_value().string_value
        stats_topic = self.get_parameter('stats_topic').get_parameter_value().string_value
        self._width = self.get_parameter('image_width').get_parameter_value().integer_value
        self._height = self.get_parameter('image_height').get_parameter_value().integer_value
        rate_hz = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        self._frames_received = 0
        self._frames_dropped = 0
        # Precompute the constant frame payload once; rebuilding a ~900k
        # element list per tick at 15 Hz would skew HIL latency measurements.
        # (rclpy accepts bytes for uint8[] fields and stores them without a
        # per-element conversion loop.)
        self._frame_data = bytes([80, 80, 80] * (self._width * self._height))
        self._frame_pub = self.create_publisher(Image, frame_topic, 10)
        self._stats_pub = self.create_publisher(UInt32, stats_topic, 10)
        self._timer = self.create_timer(1.0 / max(rate_hz, 1.0), self._publish_frame)

    def _publish_frame(self) -> None:
        # The counter increments BEFORE publishing so the stats message and
        # the frame it describes stay in lockstep (frame N <-> stats N).
        self._frames_received += 1
        stamp = self.get_clock().now().to_msg()

        frame = Image()
        frame.header.stamp = stamp
        frame.header.frame_id = 'edge_usb_camera'
        frame.height = self._height
        frame.width = self._width
        frame.encoding = 'rgb8'
        frame.is_bigendian = 0
        frame.step = self._width * 3
        frame.data = self._frame_data
        self._frame_pub.publish(frame)

        # Cumulative counter (not a delta): consumers detect drops by
        # looking for gaps between consecutive values, and a restart is
        # visible as the counter moving backwards.
        stats = UInt32()
        stats.data = self._frames_received
        self._stats_pub.publish(stats)


def main() -> None:
    rclpy.init()
    node = MockUsbCameraHilNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
