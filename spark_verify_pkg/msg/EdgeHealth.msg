# Heartbeat telemetry from the edge deployment node (Phase 4 HIL).
# Published at a fixed 2 Hz regardless of traffic, so silence itself
# signals a dead node. The HIL test suite asserts directly on this message,
# meaning production monitoring gets the same guarantees the tests verify.
std_msgs/Header header
# Latency of the most recent inference vs the configured budget.
float64 inference_latency_ms
float64 max_latency_threshold_ms
# Cumulative USB camera frame statistics (drop detection).
uint32 frames_received
uint32 frames_dropped
# Pre-digested pass/fail summaries for simple consumers and dashboards.
bool latency_ok
bool frame_pipeline_ok
# Camera optics recommendation from the lens advisor (computed at startup).
string recommended_lens
