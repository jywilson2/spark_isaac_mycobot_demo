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

"""Joint naming conventions for mock vs live (URDF) MyCobot 280 M5 bridges."""

from spark_verify_nodes.articulation_model_py import DEFAULT_JOINT_NAMES

# Revolute joints in mycobot_280_m5.urdf (Limo Cobot arm variant).
URDF_REVOLUTE_JOINTS = [
    'joint2_to_joint1',
    'joint3_to_joint2',
    'joint4_to_joint3',
    'joint5_to_joint4',
    'joint6_to_joint5',
    'joint6output_to_joint6',
]

URDF_BASE_FRAME = 'g_base'
URDF_END_EFFECTOR_FRAME = 'joint6_flange'


def mock_joint_names() -> list[str]:
    return list(DEFAULT_JOINT_NAMES)


def urdf_joint_names() -> list[str]:
    return list(URDF_REVOLUTE_JOINTS)


def map_mock_positions_to_urdf(positions: list[float]) -> list[float]:
    if len(positions) != len(DEFAULT_JOINT_NAMES):
        raise ValueError(
            f'Expected {len(DEFAULT_JOINT_NAMES)} mock joint positions, got {len(positions)}')
    return list(positions)
