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

from spark_verify_nodes.isaac_lab_mdp_env import IsaacLabMyCobotPickPlaceEnv
from spark_verify_nodes.live_sim_gate import require_live_sim, spin_until
from spark_verify_nodes.mycobot_mdp import MyCobotPickPlaceMDP
from spark_verify_nodes.reward_function import compute_total_reward
from spark_verify_nodes.safety_boundary_evaluator_py import evaluate_safety_boundaries
from spark_verify_pkg.msg import BlockDetection, RlObservation
from std_msgs.msg import Float32


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    os.environ['ROS_DOMAIN_ID'] = '42'
    launch_file = os.path.join(
        os.path.dirname(__file__), '..', 'launch', 'phase2_live_ecosystem.launch.py')

    return launch.LaunchDescription([
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(launch_file),
        ),
        launch_testing.actions.ReadyToTest(),
    ])


class TestPhase2LiveEcosystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()
        require_live_sim(
            required_topics=[
                '/mycobot/joint_states',
                '/mycobot/camera/rgb',
                '/mycobot/vision/block_detection',
            ],
            timeout_sec=8.0,
            reason='Isaac Sim live bridge not detected for Phase 2 live tests',
        )

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('phase2_live_integration_test')
        self.detections = []
        self.observations = []
        self.rewards = []
        self.safety_penalties = []
        self.node.create_subscription(
            BlockDetection,
            '/mycobot/vision/block_detection',
            self.detections.append,
            10,
        )
        self.node.create_subscription(
            RlObservation,
            '/mycobot/rl/observation',
            self.observations.append,
            10,
        )
        self.node.create_subscription(
            Float32, '/mycobot/rl/live_reward', self.rewards.append, 10)
        self.node.create_subscription(
            Float32, '/mycobot/rl/live_safety_penalty', self.safety_penalties.append, 10)

    def tearDown(self):
        self.node.destroy_node()

    def test_live_vision_pipeline_publishes_block_centroid(self):
        received = spin_until(self.node, lambda: len(self.detections) > 0, 25.0)
        self.assertTrue(received, 'Timed out waiting for live block detections')

        detection = next((item for item in reversed(self.detections) if item.detected), None)
        self.assertIsNotNone(detection, 'Live vision pipeline did not detect the workspace block')
        assert detection is not None
        self.assertGreater(detection.confidence, 0.0)
        self.assertEqual(len(detection.bbox_xyxy), 4)
        self.assertGreater(detection.centroid_x, 0.0)
        self.assertLess(detection.centroid_x, 1.0)
        self.assertGreater(detection.centroid_y, 0.0)
        self.assertLess(detection.centroid_y, 1.0)

    def test_live_rl_observation_contains_centroid_and_joint_state(self):
        obs_received = spin_until(self.node, lambda: len(self.observations) > 0, 25.0)
        self.assertTrue(obs_received, 'Timed out waiting for live RL observations')

        observation = self.observations[-1]
        self.assertEqual(observation.observation_dim, len(observation.observation))
        self.assertEqual(observation.observation_dim, MyCobotPickPlaceMDP.OBSERVATION_DIM)
        self.assertGreater(observation.observation[0], 0.0)
        self.assertLess(observation.observation[0], 1.0)
        self.assertGreater(observation.observation[1], 0.0)
        self.assertLess(observation.observation[1], 1.0)
        self.assertLess(observation.observation[2], observation.observation[4])
        self.assertLess(observation.observation[3], observation.observation[5])

    def test_live_reward_penalizes_safety_violations_before_training(self):
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

        env = IsaacLabMyCobotPickPlaceEnv()
        detection = BlockDetection()
        detection.detected = True
        detection.centroid_x = 0.5
        detection.centroid_y = 0.5
        detection.bbox_xyxy = [0.4, 0.4, 0.6, 0.6]
        safe_step = env.step_from_live_messages(
            detection,
            joint_positions=[0.0] * 6,
            joint_velocities=[0.0] * 6,
            end_effector_x=0.1,
            end_effector_y=0.0,
            end_effector_z=0.12,
            is_grasped=False,
            block_height=0.0,
        )
        unsafe_step = env.step_from_live_messages(
            detection,
            joint_positions=[3.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            joint_velocities=[0.0] * 6,
            end_effector_x=0.1,
            end_effector_y=0.0,
            end_effector_z=0.12,
            is_grasped=False,
            block_height=0.0,
        )
        self.assertFalse(safe_step.boundary_violated)
        self.assertTrue(unsafe_step.boundary_violated)
        self.assertGreater(safe_step.reward, unsafe_step.reward)

        penalized = compute_total_reward(
            centroid_x=0.5,
            centroid_y=0.5,
            end_effector_x=0.5,
            end_effector_y=0.5,
            is_grasped=True,
            block_height=0.15,
            safety_penalty=1.0,
        )
        self.assertLess(
            penalized,
            compute_total_reward(
                centroid_x=0.5,
                centroid_y=0.5,
                end_effector_x=0.5,
                end_effector_y=0.5,
                is_grasped=True,
                block_height=0.15,
                safety_penalty=0.0,
            ),
        )

        safety = evaluate_safety_boundaries(
            env.safety_config,
            [3.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0] * 6,
            0.1,
            0.0,
            0.12,
        )
        self.assertTrue(safety.boundary_violated)

        reward_received = spin_until(self.node, lambda: len(self.rewards) > 0, 25.0)
        self.assertTrue(reward_received, 'Timed out waiting for live MDP reward metrics')
