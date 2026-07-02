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

"""
Recommend Raspberry Pi USB camera optics from workspace geometry.

The lens must satisfy two constraints simultaneously:

1. Coverage: the horizontal field of view at the working distance must span
   the full manipulator workspace (plus margin) so the block can never leave
   the frame while the arm operates.
2. Resolution: the block must still subtend enough image pixels at that
   coverage for the vision tracker to produce a stable bounding box.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class WorkspaceConstraints:
    """Physical task constraints for the RPi USB camera lens selection."""

    working_distance_m: float = 0.55
    # MyCobot 280 has a 280 mm reach; the observed span is twice the reach.
    workspace_span_m: float = 0.56
    block_size_m: float = 0.04
    sensor_width_mm: float = 3.68
    image_width_px: int = 640
    required_coverage_margin: float = 1.25
    min_block_pixels: float = 24.0


def compute_block_pixel_extent(constraints: WorkspaceConstraints | None = None) -> float:
    """Return the horizontal pixel extent of the block at full workspace coverage."""
    active = constraints or WorkspaceConstraints()
    covered_span_m = active.workspace_span_m * active.required_coverage_margin
    if covered_span_m <= 0.0:
        raise ValueError('Workspace span and coverage margin must be positive')
    return active.block_size_m / covered_span_m * active.image_width_px


def recommend_camera_lens(constraints: WorkspaceConstraints | None = None) -> tuple[float, str]:
    """
    Return focal length (mm) and lens category recommendation.

    Raises ValueError if the constraints are non-physical or if no single
    fixed lens can both cover the workspace and resolve the block.
    """
    active = constraints or WorkspaceConstraints()
    if active.working_distance_m <= 0.0:
        raise ValueError('Working distance must be positive')
    if active.workspace_span_m <= 0.0 or active.block_size_m <= 0.0:
        raise ValueError('Workspace span and block size must be positive')
    if active.sensor_width_mm <= 0.0 or active.image_width_px <= 0:
        raise ValueError('Sensor width and image width must be positive')
    if active.required_coverage_margin < 1.0:
        raise ValueError('Coverage margin must be at least 1.0')

    required_horizontal_fov_rad = 2.0 * math.atan(
        (active.workspace_span_m * active.required_coverage_margin) /
        (2.0 * active.working_distance_m))
    focal_length_mm = active.sensor_width_mm / (
        2.0 * math.tan(required_horizontal_fov_rad / 2.0))

    block_pixels = compute_block_pixel_extent(active)
    if block_pixels < active.min_block_pixels:
        raise ValueError(
            f'Block subtends only {block_pixels:.1f} px at full workspace coverage; '
            f'at least {active.min_block_pixels:.1f} px are required for tracking')

    if focal_length_mm <= 2.8:
        category = 'wide_angle_m12_2_8mm'
    elif focal_length_mm <= 3.6:
        category = 'standard_m12_3_6mm'
    else:
        category = 'narrow_m12_6mm'

    return focal_length_mm, category
