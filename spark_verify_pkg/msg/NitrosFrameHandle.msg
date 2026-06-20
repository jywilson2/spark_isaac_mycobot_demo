# Mock NITROS zero-copy frame handle for Phase 1 verification.
# Mirrors the PID/FD descriptor pattern used by Isaac ROS NITROS bridge messages.
std_msgs/Header header
uint32 height
uint32 width
string encoding
uint32 step
int32[] data
string uid
uint32 device_id
