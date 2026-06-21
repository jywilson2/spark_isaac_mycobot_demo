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

"""Detect whether Isaac Sim live ROS 2 bridge telemetry is reachable."""

from __future__ import annotations

import os
import time
from typing import Callable

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


def skip_live_tests_enabled() -> bool:
    return os.environ.get('SPARK_SKIP_LIVE_SIM_TESTS', '0').lower() in {'1', 'true', 'yes'}


def force_live_tests_enabled() -> bool:
    return os.environ.get('SPARK_FORCE_LIVE_SIM_TESTS', '0').lower() in {'1', 'true', 'yes'}


class LiveSimGate:
    """Probe /clock and optional topic publishers from Isaac Sim."""

    def __init__(
        self,
        clock_topic: str = '/clock',
        required_topics: list[str] | None = None,
        timeout_sec: float = 5.0,
    ) -> None:
        self._clock_topic = clock_topic
        self._required_topics = required_topics or []
        self._timeout_sec = timeout_sec
        self._clock_received = False

    @classmethod
    def probe(
        cls,
        *,
        clock_topic: str = '/clock',
        required_topics: list[str] | None = None,
        timeout_sec: float = 5.0,
    ) -> bool:
        if skip_live_tests_enabled():
            return False
        if force_live_tests_enabled():
            return True

        gate = cls(
            clock_topic=clock_topic,
            required_topics=required_topics,
            timeout_sec=timeout_sec,
        )
        return gate.wait_until_ready()

    def wait_until_ready(self) -> bool:
        initialized_here = False
        if not rclpy.ok():
            rclpy.init()
            initialized_here = True

        node = Node('live_sim_gate_probe')
        try:
            node.create_subscription(Clock, self._clock_topic, self._on_clock, 10)
            deadline = time.time() + self._timeout_sec
            while time.time() < deadline:
                rclpy.spin_once(node, timeout_sec=0.1)
                if self._clock_received and self._topics_ready(node):
                    return True
            return False
        finally:
            node.destroy_node()
            if initialized_here:
                rclpy.shutdown()

    def _on_clock(self, _msg: Clock) -> None:
        self._clock_received = True

    def _topics_ready(self, node: Node) -> bool:
        topic_names_and_types = dict(node.get_topic_names_and_types())
        for topic in self._required_topics:
            if topic not in topic_names_and_types:
                return False
        return True


def require_live_sim(
    *,
    required_topics: list[str] | None = None,
    timeout_sec: float = 5.0,
    reason: str = 'Isaac Sim live bridge not detected',
) -> None:
    """Raise unittest.SkipTest when live sim telemetry is unavailable."""
    import unittest

    topics = required_topics or []
    if not LiveSimGate.probe(required_topics=topics, timeout_sec=timeout_sec):
        raise unittest.SkipTest(reason)


def spin_until(
    node: Node,
    predicate: Callable[[], bool],
    timeout_sec: float = 15.0,
) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return True
    return False
