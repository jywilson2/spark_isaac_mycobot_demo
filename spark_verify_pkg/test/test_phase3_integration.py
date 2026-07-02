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
# PHASE 3 VERIFICATION (spec.md): "passes mock visual tensors through the
# exported ONNX model, evaluates the driver, and asserts expected serial
# byte outputs for joint states." Two new test techniques appear here
# beyond the phase 1/2 patterns:
#
#   * FAULT INJECTION: the test itself PUBLISHES a hostile PolicyInference
#     (3.5 rad target) onto the live graph to prove the driver's safety
#     gate rejects it — the test acts as an adversarial policy.
#   * NEGATIVE ASSERTION: proving something did NOT happen (no out-of-bound
#     serial output) by scanning everything that WAS emitted.
# ---------------------------------------------------------------------------

import os
import time
import unittest

import launch
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy

from spark_verify_nodes.mock_onnx_policy import run_mock_onnx_inference
from spark_verify_pkg.msg import PolicyInference, SerialCommand


def decode_first_joint_rad(payload: list[int]) -> float:
    """
    Independently decode joint1 from a pymycobot packet (bytes 4-5).

    Deliberately re-implemented here rather than importing the encoder
    module: an independent decoder catches bugs that a shared
    implementation would mirror on both sides.
    """
    if len(payload) < 6:
        return 0.0
    # Big-endian int16 reassembly with manual two's-complement sign fix
    # (Python ints are unbounded, so values > 32767 are really negatives).
    raw = (payload[4] << 8) | payload[5]
    if raw > 32767:
        raw -= 65536
    # Wire unit is tenths of a degree; convert back to radians.
    return raw / 10.0 * 3.141592653589793 / 180.0


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    # Unique DDS domain (see phase 1 test for the isolation rationale).
    os.environ['ROS_DOMAIN_ID'] = '43'
    launch_file = os.path.join(
        os.path.dirname(__file__), '..', 'launch', 'phase3_mock_ecosystem.launch.py')

    return launch.LaunchDescription([
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(launch_file),
        ),
        launch_testing.actions.ReadyToTest(),
    ])


class TestPhase3MockEcosystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('phase3_integration_test')
        self.inferences = []
        self.serial_commands = []
        self.inference_sub = self.node.create_subscription(
            PolicyInference,
            '/mycobot/policy/inference',
            self.inferences.append,
            10,
        )
        self.serial_sub = self.node.create_subscription(
            SerialCommand,
            '/mycobot/hardware/serial_command',
            self.serial_commands.append,
            10,
        )

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

    def test_mock_onnx_inference_produces_joint_targets(self):
        # Part 1 — direct call: exercise the policy function in-process with
        # a hand-built observation (layout: 2 centroid + 4 bbox + height +
        # grasp, then 6 joints appended below).
        observation = [0.5, 0.5, 0.4, 0.4, 0.6, 0.6, 0.12, 0.0]
        observation.extend([0.4, -0.2, 0.6, -0.3, 0.5, -0.1])
        targets = run_mock_onnx_inference(observation)
        self.assertEqual(len(targets), 6)

        # Part 2 — live pipeline: the SAME function running inside
        # onnx_inference_node must also be producing accepted inferences
        # from the real observation stream.
        inference_ok = self._spin_until(lambda: len(self.inferences) > 0)
        self.assertTrue(inference_ok, 'Timed out waiting for policy inference output')
        self.assertTrue(self.inferences[-1].accepted)

    def test_driver_emits_expected_serial_bytes(self):
        serial_ok = self._spin_until(lambda: len(self.serial_commands) > 0)
        self.assertTrue(serial_ok, 'Timed out waiting for pymycobot serial output')

        # Structural assertions on the emitted packet — the "expected
        # serial byte outputs" the spec requires: framing bytes, command
        # id, minimum length, and a checksum recomputed independently.
        serial = self.serial_commands[-1]
        self.assertEqual(serial.command_id, 0x22)
        self.assertGreaterEqual(len(serial.payload), 17)
        self.assertEqual(serial.payload[0], 0xFE)
        self.assertEqual(serial.payload[1], 0xFE)
        self.assertEqual(serial.payload[3], 0x22)

        checksum = sum(serial.payload[2:-1]) & 0xFF
        self.assertEqual(checksum, serial.payload[-1])

    def test_driver_rejects_out_of_bound_inference(self):
        # FAULT INJECTION: publish a hostile inference directly onto the
        # driver's input topic. 3.5 rad (~200 deg) exceeds the 2.879793 rad
        # joint limit, and accepted=True ensures the driver's own safety
        # gate — not an upstream flag — must do the rejecting.
        pub = self.node.create_publisher(PolicyInference, '/mycobot/policy/inference', 10)
        rejected = PolicyInference()
        rejected.header.stamp = self.node.get_clock().now().to_msg()
        rejected.joint_targets_rad = [3.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        rejected.inference_latency_ms = 5.0
        rejected.accepted = True

        # Publish repeatedly with sleeps: the publisher was just created
        # and DDS discovery needs time; repetition guarantees delivery
        # while spin_once keeps our own subscriptions draining.
        for _ in range(5):
            pub.publish(rejected)
            time.sleep(0.2)
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # NEGATIVE ASSERTION: the legitimate pipeline keeps emitting serial
        # (joint1 stays near 0.45 rad), so we cannot assert "no output".
        # Instead, decode EVERY packet seen and verify none carries the
        # hostile 3.5 rad target — proving it died at the safety gate.
        for serial in self.serial_commands:
            joint0 = decode_first_joint_rad(list(serial.payload))
            self.assertLess(abs(joint0), 3.0)
