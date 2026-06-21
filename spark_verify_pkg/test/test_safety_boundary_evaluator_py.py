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

import unittest

from spark_verify_nodes.safety_boundary_evaluator_py import (
    evaluate_safety_boundaries,
    SafetyBoundaryConfig,
)


class TestSafetyBoundaryEvaluatorPy(unittest.TestCase):

    def test_joint_limit_violation_penalized(self) -> None:
        config = SafetyBoundaryConfig()
        safe = evaluate_safety_boundaries(
            config,
            joint_positions=[0.0] * 6,
            joint_velocities=[0.0] * 6,
            end_effector_x=0.1,
            end_effector_y=0.0,
            end_effector_z=0.12,
        )
        unsafe = evaluate_safety_boundaries(
            config,
            joint_positions=[3.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            joint_velocities=[0.0] * 6,
            end_effector_x=0.1,
            end_effector_y=0.0,
            end_effector_z=0.12,
        )
        self.assertFalse(safe.boundary_violated)
        self.assertTrue(unsafe.boundary_violated)
        self.assertGreater(unsafe.total_penalty, 0.0)
