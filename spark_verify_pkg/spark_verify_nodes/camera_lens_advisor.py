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

"""Recommend Raspberry Pi USB camera optics from workspace geometry."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WorkspaceConstraints:
    working_distance_m: float = 0.55
    block_size_m: float = 0.04
    sensor_width_mm: float = 3.68
    required_coverage_margin: float = 1.25


def recommend_camera_lens(constraints: WorkspaceConstraints | None = None) -> tuple[float, str]:
    """Return focal length (mm) and lens category recommendation."""
    active = constraints or WorkspaceConstraints()
    required_horizontal_fov_rad = 2.0 * math.atan(
        (active.block_size_m * active.required_coverage_margin) /
        (2.0 * active.working_distance_m))
    focal_length_mm = active.sensor_width_mm / (
        2.0 * math.tan(required_horizontal_fov_rad / 2.0))

    if focal_length_mm <= 2.8:
        category = 'wide_angle_m12_2_8mm'
    elif focal_length_mm <= 3.6:
        category = 'standard_m12_3_6mm'
    else:
        category = 'narrow_m12_6mm'

    return focal_length_mm, category
