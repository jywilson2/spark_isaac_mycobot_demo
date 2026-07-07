# Output of the block vision tracker (Phase 2).
# All coordinates are NORMALIZED to 0..1 by the image dimensions, so the RL
# observation stays independent of camera resolution.
std_msgs/Header header
# False when no block is visible; a message is still published so consumers
# can distinguish "no block" from "tracker dead".
bool detected
# Bounding box as [x_min, y_min, x_max, y_max]. Fixed-size arrays ([4]) are
# enforced by rosidl: assigning any other length raises an error.
float32[4] bbox_xyxy
float32 centroid_x
float32 centroid_y
# 1.0 for the threshold detector; a neural detector would report its score.
float32 confidence
# Block pose in the robot base frame (meters). Populated when detected=true.
float32 block_base_x
float32 block_base_y
float32 block_base_z
# Commanded EE target derived from the block pose (meters, base frame).
float32 target_ee_x
float32 target_ee_y
float32 target_ee_z
# True when block_base_* and target_ee_* are valid for motion control.
bool pose_valid
