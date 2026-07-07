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

"""Vision localization: detector + depth unprojection → block pose in robot base frame."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from isaac_lab.mdp_core import compute_target_ee_pose
from isaac_lab.phase6_red_block.block_vision import detect_red_block_in_camera_batch


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for the EE-mounted camera."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def default_ee_camera_intrinsics(
    *,
    width: int = 128,
    height: int = 128,
    focal_length_mm: float = 24.0,
    horizontal_aperture_mm: float = 20.955,
) -> CameraIntrinsics:
    """Match ``MyCobotPickPlaceEnvCfg.ee_camera`` spawn parameters."""

    focal_px = focal_length_mm / horizontal_aperture_mm * float(width)
    return CameraIntrinsics(
        fx=focal_px,
        fy=focal_px,
        cx=(float(width) - 1.0) * 0.5,
        cy=(float(height) - 1.0) * 0.5,
        width=width,
        height=height,
    )


def quat_apply_wxyz(quat_wxyz: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Rotate 3-vectors by wxyz quaternions (batch)."""

    w = quat_wxyz[:, 0:1]
    qvec = quat_wxyz[:, 1:4]
    t = 2.0 * torch.cross(qvec, vectors, dim=-1)
    return vectors + w * t + torch.cross(qvec, t, dim=-1)


def _sample_depth_at_centroid(
    depth: torch.Tensor,
    centroid_x: torch.Tensor,
    centroid_y: torch.Tensor,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    if depth.ndim == 4:
        depth_map = depth[..., 0]
    elif depth.ndim == 3:
        depth_map = depth
    else:
        raise ValueError('depth must have shape (N, H, W) or (N, H, W, 1)')

    px = torch.clamp(
        torch.round(centroid_x * float(width)).long(),
        0,
        width - 1,
    )
    py = torch.clamp(
        torch.round(centroid_y * float(height)).long(),
        0,
        height - 1,
    )
    batch = torch.arange(depth_map.shape[0], device=depth_map.device)
    sampled = depth_map[batch, py, px]
    return sampled.clamp(min=0.05, max=2.0)


def unproject_centroid_to_camera(
    centroid_x: torch.Tensor,
    centroid_y: torch.Tensor,
    depth_m: torch.Tensor,
    intrinsics: CameraIntrinsics,
) -> torch.Tensor:
    """Unproject normalized centroids to 3D points in the camera optical frame."""

    px = centroid_x * float(intrinsics.width)
    py = centroid_y * float(intrinsics.height)
    x = (px - intrinsics.cx) * depth_m / intrinsics.fx
    y = (py - intrinsics.cy) * depth_m / intrinsics.fy
    z = depth_m
    return torch.stack([x, y, z], dim=-1)


def localize_block_from_camera_batch(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    camera_pos_base: torch.Tensor,
    camera_quat_wxyz: torch.Tensor,
    *,
    intrinsics: CameraIntrinsics | None = None,
    block_half_size_m: float = 0.02,
    approach_offset_m: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detect block, unproject depth, and express pose in the robot base frame.

    Returns ``block_pos_base``, ``target_ee_base``, ``detected``, ``pose_valid``.
    The detector front-end is a color-threshold blob finder; swap for Isaac ROS
    DNN output while keeping this geometric back-end.
    """

    intrinsics = intrinsics or default_ee_camera_intrinsics(
        width=int(rgb.shape[-2]),
        height=int(rgb.shape[-3]),
    )

    centroid_x, centroid_y, detected, _bbox = detect_red_block_in_camera_batch(rgb)
    depth_m = _sample_depth_at_centroid(
        depth,
        centroid_x,
        centroid_y,
        width=intrinsics.width,
        height=intrinsics.height,
    )
    points_cam = unproject_centroid_to_camera(centroid_x, centroid_y, depth_m, intrinsics)
    points_base = quat_apply_wxyz(camera_quat_wxyz, points_cam) + camera_pos_base

    block_z = points_base[:, 2].clone()
    block_z = torch.where(detected, block_z, torch.zeros_like(block_z))
    block_pos = points_base.clone()
    block_pos[:, 2] = torch.where(
        detected,
        block_z,
        torch.zeros_like(block_z),
    )

    target_ee = torch.zeros_like(block_pos)
    for env_idx in range(block_pos.shape[0]):
        if bool(detected[env_idx].item()):
            tx, ty, tz = compute_target_ee_pose(
                float(block_pos[env_idx, 0].item()),
                float(block_pos[env_idx, 1].item()),
                float(block_pos[env_idx, 2].item()),
                block_half_size_m=block_half_size_m,
                approach_offset_m=approach_offset_m,
            )
            target_ee[env_idx, 0] = tx
            target_ee[env_idx, 1] = ty
            target_ee[env_idx, 2] = tz

    pose_valid = detected & torch.isfinite(block_pos).all(dim=-1) & torch.isfinite(target_ee).all(dim=-1)
    return block_pos, target_ee, detected, pose_valid
