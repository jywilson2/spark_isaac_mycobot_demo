# A framed pymycobot serial packet (Phases 3-4), ready to write to the
# MyCobot's serial port. In the mock ecosystem tests decode and assert on
# these bytes; on hardware a serial writer node would transmit payload as-is.
std_msgs/Header header
# Raw frame bytes: [0xFE, 0xFE, length, command, ...data..., checksum].
uint8[] payload
# Duplicates payload[3] so consumers can filter without parsing the frame.
uint8 command_id
