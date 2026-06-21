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

from spark_verify_nodes.live_sim_gate import LiveSimGate, skip_live_tests_enabled


class TestLiveSimGate(unittest.TestCase):

    def test_skip_flag_respected(self) -> None:
        previous = os.environ.get('SPARK_SKIP_LIVE_SIM_TESTS')
        os.environ['SPARK_SKIP_LIVE_SIM_TESTS'] = '1'
        try:
            self.assertTrue(skip_live_tests_enabled())
            self.assertFalse(LiveSimGate.probe(timeout_sec=0.1))
        finally:
            if previous is None:
                os.environ.pop('SPARK_SKIP_LIVE_SIM_TESTS', None)
            else:
                os.environ['SPARK_SKIP_LIVE_SIM_TESTS'] = previous
