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
# PHASE 4 VERIFICATION (spec.md): the HIL suite must verify that "AI Hat
# inference latency remains below target thresholds, the USB camera frames
# process without dropping, and the hardware serial nodes reject
# out-of-bound joint angles before transmitting physical motor commands."
#
# Note the TELEMETRY-DRIVEN style: instead of probing internals, the tests
# assert on the /edge/health heartbeat — the same signal an operator (or a
# watchdog on the real RPi) would monitor. If the health message carries
# enough information for the tests, it carries enough for production
# monitoring; the test doubles as a review of the telemetry design.
# ---------------------------------------------------------------------------

import os
import time
import unittest

import launch
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy

from spark_verify_nodes.camera_lens_advisor import recommend_camera_lens
from spark_verify_pkg.msg import EdgeHealth, PolicyInference, SerialCommand


def decode_first_joint_rad(payload: list[int]) -> float:
    """
    Independently decode joint1 from a pymycobot packet (bytes 4-5).

    Same independent-decoder rationale as the phase 3 suite: never verify
    an encoder with itself.
    """
    if len(payload) < 6:
        return 0.0
    raw = (payload[4] << 8) | payload[5]
    if raw > 32767:
        raw -= 65536
    return raw / 10.0 * 3.141592653589793 / 180.0


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    # Unique DDS domain (see phase 1 test for the isolation rationale).
    os.environ['ROS_DOMAIN_ID'] = '44'
    launch_file = os.path.join(
        os.path.dirname(__file__), '..', 'launch', 'phase4_hil_ecosystem.launch.py')

    return launch.LaunchDescription([
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(launch_file),
        ),
        launch_testing.actions.ReadyToTest(),
    ])


class TestPhase4HilEcosystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('phase4_hil_integration_test')
        self.health_reports = []
        self.edge_serial = []
        self.health_sub = self.node.create_subscription(
            EdgeHealth, '/edge/health', self.health_reports.append, 10)
        # The EDGE node's gated serial output — distinct from the Phase 3
        # driver's /mycobot/hardware/serial_command topic.
        self.serial_sub = self.node.create_subscription(
            SerialCommand, '/edge/hardware/serial_command', self.edge_serial.append, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _spin_until(self, predicate, timeout_sec=15.0):
        # Standard poll-with-timeout idiom (see phase 1 test for details).
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if predicate():
                return True
        return False

    def test_camera_lens_recommendation_is_computed(self):
        # In-process check of the optics math (no topics needed): a
        # physically plausible focal length in one of the commodity M12
        # lens categories. Exact-value tests live in
        # test_camera_lens_advisor.py; this is the integration smoke check.
        focal_length, category = recommend_camera_lens()
        self.assertGreater(focal_length, 0.0)
        self.assertIn(category, {
            'wide_angle_m12_2_8mm',
            'standard_m12_3_6mm',
            'narrow_m12_6mm',
        })

    def test_edge_health_reports_latency_and_frame_pipeline(self):
        # The first health tick can fire before the first camera stats message
        # arrives, so wait for a report that has observed camera traffic.
        # (Waiting merely for len > 0 would be a race condition.)
        received = self._spin_until(
            lambda: any(report.frames_received > 0 for report in self.health_reports))
        self.assertTrue(received, 'Timed out waiting for edge health reports with frames')

        health = next(
            report for report in reversed(self.health_reports) if report.frames_received > 0)
        # Spec threshold checks, straight off the telemetry: latency below
        # budget, and a lossless frame pipeline.
        self.assertLessEqual(health.inference_latency_ms, health.max_latency_threshold_ms)
        self.assertTrue(health.latency_ok)
        self.assertTrue(health.frame_pipeline_ok)
        self.assertGreater(health.frames_received, 0)
        self.assertEqual(health.frames_dropped, 0)
        self.assertGreater(len(health.recommended_lens), 0)

    def test_edge_node_rejects_out_of_bound_joint_angles(self):
        # FAULT INJECTION against the EDGE gate this time (the phase 3 suite
        # already covered the C++ driver's gate — defense in depth needs
        # each layer proven independently). 3.5 rad > the node's 2.8 rad
        # max_joint_target_rad parameter.
        pub = self.node.create_publisher(PolicyInference, '/mycobot/policy/inference', 10)
        rejected = PolicyInference()
        rejected.header.stamp = self.node.get_clock().now().to_msg()
        rejected.joint_targets_rad = [3.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        rejected.inference_latency_ms = 5.0
        rejected.accepted = True

        # Repeat-publish to ride out DDS discovery of the new publisher.
        for _ in range(5):
            pub.publish(rejected)
            time.sleep(0.2)
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Negative assertion: decode every packet the edge node emitted and
        # confirm the hostile target never reached the serial topic.
        for serial in self.edge_serial:
            joint0 = decode_first_joint_rad(list(serial.payload))
            self.assertLess(abs(joint0), 3.0)
