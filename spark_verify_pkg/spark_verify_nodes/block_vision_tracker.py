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

"""Mock block detector that converts RGB frames into bounding boxes and centroids."""

from typing import Optional, Tuple

from sensor_msgs.msg import Image
from spark_verify_pkg.msg import BlockDetection


def detect_block_in_rgb(
    image: Image,
    red_threshold: int = 180,
) -> Optional[Tuple[list[float], float, float]]:
    if image.encoding not in {'rgb8', 'bgr8'}:
        return None

    width = int(image.width)
    height = int(image.height)
    if width <= 0 or height <= 0 or len(image.data) < width * height * 3:
        return None

    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    red_pixels = 0

    for row in range(height):
        row_offset = row * image.step
        for col in range(width):
            pixel_offset = row_offset + col * 3
            red = image.data[pixel_offset]
            green = image.data[pixel_offset + 1]
            blue = image.data[pixel_offset + 2]
            if red >= red_threshold and red > green + 30 and red > blue + 30:
                red_pixels += 1
                min_x = min(min_x, col)
                min_y = min(min_y, row)
                max_x = max(max_x, col)
                max_y = max(max_y, row)

    if red_pixels == 0 or max_x < min_x or max_y < min_y:
        return None

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
    message = BlockDetection()
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
    message.confidence = 1.0
    return message
