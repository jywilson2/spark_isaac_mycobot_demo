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
import time
import unittest

import launch
import launch_testing.actions
import launch_testing.markers
import pytest
import rclpy

from spark_verify_nodes.mycobot_mdp import MyCobotPickPlaceMDP
from spark_verify_nodes.reward_function import compute_total_reward
from spark_verify_pkg.msg import BlockDetection, RlObservation


EXPECTED_CENTROID_X = 0.5
EXPECTED_CENTROID_Y = 0.5
CENTROID_TOLERANCE = 0.05


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    os.environ['ROS_DOMAIN_ID'] = '42'
    launch_file = os.path.join(
        os.path.dirname(__file__), '..', 'launch', 'phase2_mock_ecosystem.launch.py')

    return launch.LaunchDescription([
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(launch_file),
        ),
        launch_testing.actions.ReadyToTest(),
    ])


class TestPhase2MockEcosystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('phase2_integration_test')
        self.detections = []
        self.observations = []
        self.detection_sub = self.node.create_subscription(
            BlockDetection,
            '/mycobot/vision/block_detection',
            self.detections.append,
            10,
        )
        self.observation_sub = self.node.create_subscription(
            RlObservation,
            '/mycobot/rl/observation',
            self.observations.append,
            10,
        )

    def tearDown(self):
        self.node.destroy_node()

    def _spin_until(self, predicate, timeout_sec=15.0):
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if predicate():
                return True
        return False

    def test_vision_pipeline_publishes_block_centroid(self):
        received = self._spin_until(lambda: len(self.detections) > 0)
        self.assertTrue(received, 'Timed out waiting for block detections')

        detection = self.detections[-1]
        self.assertTrue(detection.detected)
        self.assertAlmostEqual(detection.centroid_x, EXPECTED_CENTROID_X, delta=CENTROID_TOLERANCE)
        self.assertAlmostEqual(detection.centroid_y, EXPECTED_CENTROID_Y, delta=CENTROID_TOLERANCE)
        self.assertEqual(len(detection.bbox_xyxy), 4)
        self.assertLess(detection.bbox_xyxy[0], detection.bbox_xyxy[2])
        self.assertLess(detection.bbox_xyxy[1], detection.bbox_xyxy[3])

    def test_rl_observation_contains_centroid_and_joint_state(self):
        obs_received = self._spin_until(lambda: len(self.observations) > 0)
        self.assertTrue(obs_received, 'Timed out waiting for RL observations')

        observation = self.observations[-1]
        self.assertEqual(observation.observation_dim, len(observation.observation))
        self.assertEqual(observation.observation_dim, MyCobotPickPlaceMDP.OBSERVATION_DIM)
        self.assertAlmostEqual(
            observation.observation[0], EXPECTED_CENTROID_X, delta=CENTROID_TOLERANCE)
        self.assertAlmostEqual(
            observation.observation[1], EXPECTED_CENTROID_Y, delta=CENTROID_TOLERANCE)
        self.assertLess(observation.observation[2], observation.observation[4])
        self.assertLess(observation.observation[3], observation.observation[5])

    def test_reward_penalizes_safety_violations_before_training(self):
        mdp = MyCobotPickPlaceMDP()
        safe_reward = mdp.compute_reward(
            centroid_x=0.5,
            centroid_y=0.5,
            end_effector_x=0.5,
            end_effector_y=0.5,
            is_grasped=False,
            block_height=0.0,
            safety_penalty=0.0,
        )
        unsafe_reward = mdp.compute_reward(
            centroid_x=0.5,
            centroid_y=0.5,
            end_effector_x=0.5,
            end_effector_y=0.5,
            is_grasped=False,
            block_height=0.0,
            safety_penalty=2.0,
        )
        self.assertGreater(safe_reward, unsafe_reward)
        self.assertLess(unsafe_reward, safe_reward)

        penalized = compute_total_reward(
            centroid_x=0.5,
            centroid_y=0.5,
            end_effector_x=0.5,
            end_effector_y=0.5,
            is_grasped=True,
            block_height=0.15,
            safety_penalty=1.0,
        )
        self.assertLess(penalized, compute_total_reward(
            centroid_x=0.5,
            centroid_y=0.5,
            end_effector_x=0.5,
            end_effector_y=0.5,
            is_grasped=True,
            block_height=0.15,
            safety_penalty=0.0,
        ))
