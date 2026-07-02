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
Mock block detector that converts RGB frames into bounding boxes and centroids.

VISION BACKGROUND: This is deliberately the simplest possible detector — a
per-pixel red-color threshold followed by a bounding box over the matching
pixels. A production pipeline would use Isaac ROS GPU-accelerated detection
(e.g. DNN inference over NITROS buffers). What matters for verification is
the OUTPUT CONTRACT, not the algorithm: downstream RL code receives
normalized bbox + centroid coordinates, and this module guarantees exact,
deterministic values for the synthetic frames the mock camera renders.

These are plain functions (no Node class) so they can be unit-tested without
ROS running; block_vision_tracker_node.py wraps them into a subscriber.
"""

from typing import Optional, Tuple

from sensor_msgs.msg import Image
from spark_verify_pkg.msg import BlockDetection


def detect_block_in_rgb(
    image: Image,
    red_threshold: int = 180,
) -> Optional[Tuple[list[float], float, float]]:
    """Scan an Image message for red pixels; return (bbox, cx, cy) or None."""
    # sensor_msgs/Image stores raw bytes plus metadata describing them.
    # 'encoding' tells consumers how to interpret the bytes; this mock only
    # handles 8-bit 3-channel layouts.
    if image.encoding not in {'rgb8', 'bgr8'}:
        return None

    width = int(image.width)
    height = int(image.height)
    # Defensive validation: a malformed message (data shorter than
    # width*height*3) would otherwise cause IndexError mid-scan.
    if width <= 0 or height <= 0 or len(image.data) < width * height * 3:
        return None

    # Track the extremes of matching pixels to form the bounding box.
    # Starting min at the far edge and max at -1 means "nothing found yet".
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    red_pixels = 0

    for row in range(height):
        # 'step' is the stride in BYTES between the start of consecutive
        # rows. It can exceed width*3 when rows are padded for alignment,
        # so always compute offsets from step, never from width alone.
        row_offset = row * image.step
        for col in range(width):
            pixel_offset = row_offset + col * 3
            red = image.data[pixel_offset]
            green = image.data[pixel_offset + 1]
            blue = image.data[pixel_offset + 2]
            # "Red enough" = bright red channel AND clearly above the other
            # channels; the relative check rejects white/gray pixels where
            # all three channels are high.
            if red >= red_threshold and red > green + 30 and red > blue + 30:
                red_pixels += 1
                min_x = min(min_x, col)
                min_y = min(min_y, row)
                max_x = max(max_x, col)
                max_y = max(max_y, row)

    if red_pixels == 0 or max_x < min_x or max_y < min_y:
        return None

    # Normalize everything to 0..1 by dividing by the image dimensions.
    # Normalized coordinates make the downstream RL observation independent
    # of camera resolution — the same policy works at 640x480 or 1920x1080.
    # The +1 converts the inclusive max pixel index to an exclusive edge.
    bbox = [
        min_x / width,
        min_y / height,
        (max_x + 1) / width,
        (max_y + 1) / height,
    ]
    centroid_x = ((min_x + max_x + 1) / 2.0) / width
    centroid_y = ((min_y + max_y + 1) / 2.0) / height
    return bbox, centroid_x, centroid_y


def to_block_detection(image: Image, red_threshold: int = 180) -> BlockDetection:
    """
    Wrap the detector output in the package's BlockDetection message.

    Note that a message is ALWAYS returned, with detected=False when nothing
    was found. Publishing explicit "no detection" messages (instead of going
    silent) lets consumers distinguish "camera sees no block" from "vision
    node is dead" — an important liveness pattern in robot systems.
    """
    message = BlockDetection()
    # Copying the image header propagates the FRAME'S timestamp and frame_id
    # so downstream consumers know exactly which camera instant produced
    # this detection (essential for fusing with other sensor data).
    message.header = image.header
    detection = detect_block_in_rgb(image, red_threshold=red_threshold)
    if detection is None:
        message.detected = False
        message.bbox_xyxy = [0.0, 0.0, 0.0, 0.0]
        message.centroid_x = 0.0
        message.centroid_y = 0.0
        message.confidence = 0.0
        return message

    bbox, centroid_x, centroid_y = detection
    message.detected = True
    message.bbox_xyxy = bbox
    message.centroid_x = centroid_x
    message.centroid_y = centroid_y
    # The threshold detector is binary, so confidence is 1.0; a neural
    # detector would report its softmax score here.
    message.confidence = 1.0
    return message
