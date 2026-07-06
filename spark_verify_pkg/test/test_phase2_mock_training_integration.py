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

import json
import os
from pathlib import Path
import tempfile
import unittest

import launch
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy

from spark_verify_nodes.live_sim_gate import spin_until
from std_msgs.msg import String


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    os.environ['ROS_DOMAIN_ID'] = '42'
    os.environ['SPARK_REPO_ROOT'] = tempfile.mkdtemp(prefix='spark_mock_train_')
    launch_file = os.path.join(
        os.path.dirname(__file__), '..', 'launch', 'phase2_mock_training.launch.py')

    return launch.LaunchDescription([
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(launch_file),
        ),
        launch_testing.actions.ReadyToTest(),
    ])


class TestPhase2MockTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('phase2_mock_training_test')
        self.status_messages: list[str] = []
        self.node.create_subscription(
            String,
            '/mycobot/rl/training_status',
            lambda msg: self.status_messages.append(msg.data),
            10,
        )

    def tearDown(self):
        self.node.destroy_node()

    def test_mock_training_completes_and_writes_checkpoint(self):
        received = spin_until(
            self.node,
            lambda: any('episodes' in msg for msg in self.status_messages),
            timeout_sec=45.0,
        )
        self.assertTrue(received, 'Timed out waiting for training completion status')

        final_status = next(msg for msg in reversed(self.status_messages) if 'episodes' in msg)
        summary = json.loads(final_status)
        self.assertGreaterEqual(summary['episodes'], 2)
        self.assertGreater(summary['total_steps'], 0)

        checkpoint = Path(summary['checkpoint'])
        self.assertTrue(checkpoint.is_file())
        self.assertTrue(checkpoint.with_suffix('.npz').is_file())

        export_path = Path(summary['onnx_ready_export'])
        self.assertTrue(export_path.is_file())
