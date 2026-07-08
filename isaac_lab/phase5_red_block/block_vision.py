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

"""Batch red-block detection from EE-mounted camera RGB tensors."""

from __future__ import annotations

import torch


def _normalize_rgb(rgb: torch.Tensor) -> torch.Tensor:
    values = rgb.float()
    if values.numel() > 0 and values.max() > 1.5:
        values = values / 255.0
    return values.clamp(0.0, 1.0)


def detect_red_block_in_camera_batch(
    rgb: torch.Tensor,
    *,
    red_threshold: float = 0.55,
    channel_margin: float = 0.12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detect the red pick block in batched camera RGB frames.

    Returns ``centroid_x``, ``centroid_y`` (normalized 0–1), ``detected`` (bool),
    and ``bbox_xyxy`` (N, 4) per environment.
    """

    if rgb.ndim != 4:
        raise ValueError('rgb must have shape (num_envs, height, width, channels)')

    values = _normalize_rgb(rgb)
    red = values[..., 0]
    green = values[..., 1]
    blue = values[..., 2]
    mask = (red >= red_threshold) & (red > green + channel_margin) & (red > blue + channel_margin)

    height = mask.shape[-2]
    width = mask.shape[-1]
    coord_dtype = red.dtype
    ys = torch.arange(height, device=mask.device, dtype=coord_dtype).view(1, height, 1)
    xs = torch.arange(width, device=mask.device, dtype=coord_dtype).view(1, 1, width)

    detected = mask.any(dim=(-2, -1))
    pixel_count = mask.sum(dim=(-2, -1)).clamp(min=1.0)

    sum_x = (mask * xs).sum(dim=(-2, -1))
    sum_y = (mask * ys).sum(dim=(-2, -1))
    centroid_x = torch.where(detected, sum_x / pixel_count / float(width), torch.zeros_like(sum_x))
    centroid_y = torch.where(detected, sum_y / pixel_count / float(height), torch.zeros_like(sum_y))

    large = torch.full_like(xs, float(width))
    large_y = torch.full_like(ys, float(height))
    min_x = torch.where(mask, xs, large).amin(dim=(-2, -1))
    min_y = torch.where(mask, ys, large_y).amin(dim=(-2, -1))
    max_x = torch.where(mask, xs, torch.zeros_like(xs)).amax(dim=(-2, -1))
    max_y = torch.where(mask, ys, torch.zeros_like(ys)).amax(dim=(-2, -1))

    bbox = torch.stack(
        [
            min_x / float(width),
            min_y / float(height),
            (max_x + 1.0) / float(width),
            (max_y + 1.0) / float(height),
        ],
        dim=-1,
    )
    bbox = torch.where(detected.unsqueeze(-1), bbox, torch.zeros_like(bbox))
    return centroid_x, centroid_y, detected, bbox
