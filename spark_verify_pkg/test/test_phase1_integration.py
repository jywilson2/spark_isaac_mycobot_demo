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

import math
import os
import time
import unittest

import launch
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from spark_verify_pkg.msg import NitrosFrameHandle
from tf2_ros import Buffer, TransformListener


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    os.environ['ROS_DOMAIN_ID'] = '41'
    launch_file = os.path.join(
        os.path.dirname(__file__), '..', 'launch', 'phase1_mock_ecosystem.launch.py')

    return launch.LaunchDescription([
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(launch_file),
        ),
        launch_testing.actions.ReadyToTest(),
    ])


class TestPhase1MockEcosystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('phase1_integration_test')
        self.joint_states = []
        self.rgb_frames = []
        self.nitros_frames = []
        self.joint_state_sub = self.node.create_subscription(
            JointState, '/mycobot/joint_states', self.joint_states.append, 10)
        self.rgb_sub = self.node.create_subscription(
            Image, '/mycobot/camera/rgb', self.rgb_frames.append, 10)
        self.nitros_sub = self.node.create_subscription(
            NitrosFrameHandle, '/mycobot/camera/nitros/rgb', self.nitros_frames.append, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node, qos=QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE))

    def tearDown(self):
        self.node.destroy_node()

    def _spin_until(self, predicate, timeout_sec=15.0):
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if predicate():
                return True
        return False

    def test_joint_commands_update_articulation_state(self):
        expected = [0.4, -0.2, 0.6, -0.3, 0.5, -0.1]
        received = self._spin_until(lambda: len(self.joint_states) > 0)
        self.assertTrue(received, 'Timed out waiting for /mycobot/joint_states')

        latest = self.joint_states[-1]
        self.assertEqual(len(latest.position), 6)
        for actual, target in zip(latest.position, expected):
            self.assertAlmostEqual(actual, target, places=3)

        tf_ready = self._spin_until(
            lambda: self.tf_buffer.can_transform(
                'base_link', 'link1', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)))
        self.assertTrue(tf_ready, 'Timed out waiting for articulation TF')

        transform = self.tf_buffer.lookup_transform('base_link', 'link1', rclpy.time.Time())
        expected_x = 0.1315 * math.cos(expected[0])
        expected_y = 0.1315 * math.sin(expected[0])
        self.assertAlmostEqual(transform.transform.translation.x, expected_x, places=3)
        self.assertAlmostEqual(transform.transform.translation.y, expected_y, places=3)

    def test_camera_publishes_rgb_and_nitros_zero_copy_frames(self):
        rgb_ok = self._spin_until(lambda: len(self.rgb_frames) > 0)
        nitros_ok = self._spin_until(lambda: len(self.nitros_frames) > 0)
        self.assertTrue(rgb_ok, 'Timed out waiting for RGB camera frames')
        self.assertTrue(nitros_ok, 'Timed out waiting for NITROS frame handles')

        rgb = self.rgb_frames[-1]
        self.assertGreater(rgb.width, 0)
        self.assertGreater(rgb.height, 0)
        self.assertGreater(len(rgb.data), 0)

        nitros = self.nitros_frames[-1]
        self.assertEqual(nitros.encoding, 'rgb8')
        self.assertGreater(len(nitros.uid), 0)
        self.assertGreaterEqual(
            len(nitros.data), 3, 'NITROS handle must include PID, FD, and size')
        pid, fd, payload_size = nitros.data[0], nitros.data[1], nitros.data[2]
        self.assertGreater(pid, 0)
        self.assertGreaterEqual(fd, 0)
        self.assertEqual(payload_size, nitros.step * nitros.height)
