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

import os
import unittest

import launch
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from spark_verify_nodes.live_sim_gate import require_live_sim, spin_until
from spark_verify_pkg.msg import NitrosFrameHandle
from tf2_ros import Buffer, TransformListener


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    os.environ['ROS_DOMAIN_ID'] = '42'
    launch_file = os.path.join(
        os.path.dirname(__file__), '..', 'launch', 'phase1_live_ecosystem.launch.py')

    return launch.LaunchDescription([
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(launch_file),
        ),
        launch_testing.actions.ReadyToTest(),
    ])


class TestPhase1LiveEcosystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        require_live_sim(
            required_topics=[
                '/mycobot/joint_states',
                '/mycobot/camera/rgb',
            ],
            timeout_sec=8.0,
            reason='Isaac Sim live bridge not detected for Phase 1 live tests',
        )

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('phase1_live_integration_test')
        self.joint_states = []
        self.rgb_frames = []
        self.nitros_frames = []
        self.node.create_subscription(
            JointState, '/mycobot/joint_states', self.joint_states.append, 10)
        self.node.create_subscription(
            Image, '/mycobot/camera/rgb', self.rgb_frames.append, 10)
        self.node.create_subscription(
            NitrosFrameHandle, '/mycobot/camera/nitros/rgb', self.nitros_frames.append, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node, qos=QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE))

    def tearDown(self):
        self.node.destroy_node()

    def test_joint_commands_update_live_articulation_state(self):
        received_initial = spin_until(self.node, lambda: len(self.joint_states) > 0, 20.0)
        self.assertTrue(received_initial, 'Timed out waiting for live /mycobot/joint_states')
        initial = list(self.joint_states[-1].position)

        received_update = spin_until(
            self.node,
            lambda: len(self.joint_states) > 1 and self.joint_states[-1].position != initial,
            20.0,
        )
        self.assertTrue(
            received_update,
            'Live joint states did not change after /mycobot/joint_commands dispatch',
        )
        latest = self.joint_states[-1]
        self.assertGreaterEqual(len(latest.position), 6)

        tf_ready = spin_until(
            self.node,
            lambda: self.tf_buffer.can_transform(
                'g_base', 'joint6_flange', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)),
            10.0,
        )
        if not tf_ready:
            tf_ready = spin_until(
                self.node,
                lambda: self.tf_buffer.can_transform(
                    'base_link', 'link6', rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.2)),
                5.0,
            )
        self.assertTrue(tf_ready, 'Timed out waiting for live articulation TF')

    def test_live_camera_publishes_rgb_and_nitros_zero_copy_frames(self):
        rgb_ok = spin_until(self.node, lambda: len(self.rgb_frames) > 0, 20.0)
        nitros_ok = spin_until(self.node, lambda: len(self.nitros_frames) > 0, 20.0)
        self.assertTrue(rgb_ok, 'Timed out waiting for live RGB camera frames')
        self.assertTrue(nitros_ok, 'Timed out waiting for live NITROS frame handles')

        rgb = self.rgb_frames[-1]
        self.assertGreater(rgb.width, 0)
        self.assertGreater(rgb.height, 0)
        self.assertGreater(len(rgb.data), 0)

        nitros = self.nitros_frames[-1]
        self.assertEqual(nitros.encoding, rgb.encoding)
        self.assertGreater(len(nitros.uid), 0)
        self.assertGreaterEqual(len(nitros.data), 3)
        self.assertGreater(nitros.data[2], 0)
