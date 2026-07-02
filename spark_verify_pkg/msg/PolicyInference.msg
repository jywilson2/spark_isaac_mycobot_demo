# One policy inference result (Phase 3): the ONNX stage's output, consumed
# by the pymycobot driver and the Phase 4 edge deployment node.
# The header is propagated from the source observation so the original
# perception timestamp survives the whole pipeline (latency measurement).
std_msgs/Header header
# Fixed-size [6]: a command must always address every MyCobot joint.
float32[6] joint_targets_rad
# Wall-clock inference duration; the edge node gates on this against its
# max_inference_latency_ms budget.
float64 inference_latency_ms
# Producer-side validity flag; downstream safety gates may still veto.
bool accepted
