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

# ---------------------------------------------------------------------------
# ROS 2 FUNDAMENTALS — launch_testing: This file is BOTH a launch file and a
# test module. The launch_testing framework (registered in CMakeLists.txt
# via add_launch_test) works in two parts:
#
#   1. generate_test_description() starts the system under test — here the
#      full Phase 1 mock ecosystem — as real, separate OS processes.
#   2. The unittest.TestCase classes then run INSIDE the test process,
#      creating their own rclpy node that joins the same ROS graph to
#      observe topics and assert on live behaviour.
#
# This is integration testing at the process boundary: it exercises DDS
# discovery, QoS matching, serialization — everything unit tests skip.
# ---------------------------------------------------------------------------

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
@launch_testing.markers.keep_alive  # keep processes running across all test methods
def generate_test_description():
    # ROS_DOMAIN_ID partitions DDS discovery: only nodes with the same
    # domain see each other. Each phase's test uses a unique id (41-44) so
    # concurrently- or successively-running test suites cannot cross-talk.
    os.environ['ROS_DOMAIN_ID'] = '41'
    launch_file = os.path.join(
        os.path.dirname(__file__), '..', 'launch', 'phase1_mock_ecosystem.launch.py')

    return launch.LaunchDescription([
        # Reuse the production launch description verbatim — the test runs
        # exactly what an operator would run, not a special test topology.
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(launch_file),
        ),
        # ReadyToTest() tells the framework it may start executing the
        # unittest cases (the nodes may still be finishing discovery, which
        # is why the cases poll with timeouts instead of asserting at once).
        launch_testing.actions.ReadyToTest(),
    ])


class TestPhase1MockEcosystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # One rclpy context for the whole class; per-test contexts would
        # pay DDS participant creation cost (100+ ms) repeatedly.
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        # A fresh OBSERVER node per test method. Appending each received
        # message to a plain list is the simplest possible capture pattern:
        # assertions then inspect the list contents.
        self.node = rclpy.create_node('phase1_integration_test')
        self.joint_states = []
        self.rgb_frames = []
        self.nitros_frames = []
        # `.append` (bound method) IS the subscription callback — every
        # incoming message lands in the corresponding list.
        self.joint_state_sub = self.node.create_subscription(
            JointState, '/mycobot/joint_states', self.joint_states.append, 10)
        self.rgb_sub = self.node.create_subscription(
            Image, '/mycobot/camera/rgb', self.rgb_frames.append, 10)
        self.nitros_sub = self.node.create_subscription(
            NitrosFrameHandle, '/mycobot/camera/nitros/rgb', self.nitros_frames.append, 10)
        # TF plumbing: the TransformListener subscribes to /tf behind the
        # scenes and fills the Buffer, which is then queried like a
        # database of frame relationships.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node, qos=QoSProfile(
            depth=10, reliability=ReliabilityPolicy.RELIABLE))

    def tearDown(self):
        self.node.destroy_node()

    def _spin_until(self, predicate, timeout_sec=15.0):
        # Poll-with-timeout is THE core idiom for asynchronous assertions:
        # spin_once processes at most one callback (or times out after
        # 0.1 s), then the predicate is re-checked. Returning bool (instead
        # of asserting here) lets callers attach a descriptive message.
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if predicate():
                return True
        return False

    def test_joint_commands_update_articulation_state(self):
        # Must match the dispatcher's 'target_positions' parameter in
        # phase1_mock_ecosystem.launch.py — the test closes the loop:
        # dispatcher publishes -> bridge applies -> this node observes.
        expected = [0.4, -0.2, 0.6, -0.3, 0.5, -0.1]
        received = self._spin_until(lambda: len(self.joint_states) > 0)
        self.assertTrue(received, 'Timed out waiting for /mycobot/joint_states')

        latest = self.joint_states[-1]
        self.assertEqual(len(latest.position), 6)
        for actual, target in zip(latest.position, expected):
            self.assertAlmostEqual(actual, target, places=3)

        # Verify the TF side too: can_transform is polled because TF data
        # accumulates asynchronously as the listener receives broadcasts.
        tf_ready = self._spin_until(
            lambda: self.tf_buffer.can_transform(
                'base_link', 'link1', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)))
        self.assertTrue(tf_ready, 'Timed out waiting for articulation TF')

        # Time() with no arguments means "latest available" in tf2 queries.
        transform = self.tf_buffer.lookup_transform('base_link', 'link1', rclpy.time.Time())
        # Independent forward-kinematics recomputation of link1's expected
        # position (offset 0.1315 m rotated by joint1 = 0.4 rad).
        expected_x = 0.1315 * math.cos(expected[0])
        expected_y = 0.1315 * math.sin(expected[0])
        self.assertAlmostEqual(transform.transform.translation.x, expected_x, places=3)
        self.assertAlmostEqual(transform.transform.translation.y, expected_y, places=3)

    def test_camera_publishes_rgb_and_nitros_zero_copy_frames(self):
        # Liveness first: both camera streams must actually produce data.
        rgb_ok = self._spin_until(lambda: len(self.rgb_frames) > 0)
        nitros_ok = self._spin_until(lambda: len(self.nitros_frames) > 0)
        self.assertTrue(rgb_ok, 'Timed out waiting for RGB camera frames')
        self.assertTrue(nitros_ok, 'Timed out waiting for NITROS frame handles')

        rgb = self.rgb_frames[-1]
        self.assertGreater(rgb.width, 0)
        self.assertGreater(rgb.height, 0)
        self.assertGreater(len(rgb.data), 0)

        # The NITROS zero-copy contract: the handle must describe a mappable
        # buffer — owner PID, file descriptor, and a size consistent with
        # the advertised geometry (step x height bytes).
        nitros = self.nitros_frames[-1]
        self.assertEqual(nitros.encoding, 'rgb8')
        self.assertGreater(len(nitros.uid), 0)
        self.assertGreaterEqual(
            len(nitros.data), 3, 'NITROS handle must include PID, FD, and size')
        pid, fd, payload_size = nitros.data[0], nitros.data[1], nitros.data[2]
        self.assertGreater(pid, 0)
        self.assertGreaterEqual(fd, 0)
        self.assertEqual(payload_size, nitros.step * nitros.height)
