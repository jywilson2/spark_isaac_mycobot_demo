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
# TESTING FUNDAMENTALS — PURE PYTEST UNIT TESTS: Unlike the launch_testing
# suites, this file never touches ROS: the lens advisor is plain math, so it
# is tested as plain Python (registered in CMakeLists.txt with
# ament_add_pytest_test). The recurring technique below is INDEPENDENT
# RECOMPUTATION: each test re-derives the expected value from the pinhole
# camera model itself rather than calling the code under test, so a shared
# bug cannot cancel out.
# ---------------------------------------------------------------------------

import math

import pytest

from spark_verify_nodes.camera_lens_advisor import (
    compute_block_pixel_extent,
    recommend_camera_lens,
    WorkspaceConstraints,
)


def expected_focal_length_mm(constraints: WorkspaceConstraints) -> float:
    # Reference implementation of f = w / (2 * tan(FOV/2)) with
    # tan(FOV/2) = (span * margin / 2) / distance — written independently
    # of the production code on purpose.
    fov_rad = 2.0 * math.atan(
        (constraints.workspace_span_m * constraints.required_coverage_margin) /
        (2.0 * constraints.working_distance_m))
    return constraints.sensor_width_mm / (2.0 * math.tan(fov_rad / 2.0))


def test_default_recommendation_matches_workspace_coverage_math():
    constraints = WorkspaceConstraints()
    focal_length, category = recommend_camera_lens(constraints)

    assert focal_length == pytest.approx(expected_focal_length_mm(constraints), rel=1e-9)
    assert category == 'standard_m12_3_6mm'


def test_focal_length_is_physically_plausible_for_m12_lenses():
    focal_length, _ = recommend_camera_lens()
    assert 1.0 < focal_length < 6.5


def test_wider_workspace_recommends_wider_lens():
    # Monotonicity property: covering a wider workspace from the same
    # distance requires a wider FOV, i.e. a SHORTER focal length. Both
    # cases relax min_block_pixels because extreme spans shrink the
    # block's pixel footprint below the default tracking floor.
    wide = WorkspaceConstraints(workspace_span_m=1.2, min_block_pixels=10.0)
    narrow = WorkspaceConstraints(workspace_span_m=0.3, min_block_pixels=10.0)

    wide_focal, wide_category = recommend_camera_lens(wide)
    narrow_focal, narrow_category = recommend_camera_lens(narrow)

    assert wide_focal < narrow_focal
    assert wide_category == 'wide_angle_m12_2_8mm'
    assert narrow_category == 'narrow_m12_6mm'


def test_block_pixel_extent_supports_vision_tracking_at_full_coverage():
    constraints = WorkspaceConstraints()
    block_pixels = compute_block_pixel_extent(constraints)

    covered_span = constraints.workspace_span_m * constraints.required_coverage_margin
    expected = constraints.block_size_m / covered_span * constraints.image_width_px
    assert block_pixels == pytest.approx(expected, rel=1e-9)
    assert block_pixels >= constraints.min_block_pixels


def test_unresolvable_block_raises_value_error():
    # A 2 mm block across a 0.7 m frame subtends < 2 px at 640 wide — no
    # lens choice can fix that, so the advisor must refuse rather than
    # recommend optics that would silently break the vision tracker.
    tiny_block = WorkspaceConstraints(block_size_m=0.002)
    with pytest.raises(ValueError, match='px'):
        recommend_camera_lens(tiny_block)


# parametrize runs the same test body once per listed input — one compact
# definition yields six independently-reported test cases.
@pytest.mark.parametrize('constraints', [
    WorkspaceConstraints(working_distance_m=0.0),
    WorkspaceConstraints(workspace_span_m=-0.1),
    WorkspaceConstraints(block_size_m=0.0),
    WorkspaceConstraints(sensor_width_mm=0.0),
    WorkspaceConstraints(image_width_px=0),
    WorkspaceConstraints(required_coverage_margin=0.9),
])
def test_non_physical_constraints_raise_value_error(constraints):
    with pytest.raises(ValueError):
        recommend_camera_lens(constraints)
