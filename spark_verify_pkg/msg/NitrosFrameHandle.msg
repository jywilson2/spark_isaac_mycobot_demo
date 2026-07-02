# Mock NITROS zero-copy frame handle for Phase 1 verification.
# Mirrors the PID/FD descriptor pattern used by Isaac ROS NITROS bridge messages.
#
# BACKGROUND: NITROS (NVIDIA Isaac Transport for ROS) avoids copying image
# payloads through the DDS middleware. Instead of the pixels, the publisher
# sends a small HANDLE describing a shared/GPU memory buffer; the subscriber
# maps that buffer directly. This mock message reproduces the handle's shape
# so tests can verify the contract without real shared memory.
#
# .msg SYNTAX NOTE: each line is "<type> <field_name>"; rosidl generates a
# C++ struct and Python class from these declarations at build time.
std_msgs/Header header
uint32 height
uint32 width
string encoding
uint32 step
# Descriptor triplet: [owner process id, file descriptor, buffer size bytes]
int32[] data
# Unique buffer identity, letting consumers detect stale/reused handles.
string uid
# GPU device the buffer lives on (0 = first device).
uint32 device_id
