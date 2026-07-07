# Flattened RL observation vector (Phase 2), the policy network's input.
# Motion-policy layout:
#   [0-2] EE-to-target delta, [3] target_valid, [4] in_contact, [5-10] joints. Total: 11.
std_msgs/Header header
# Redundant with len(observation) but lets consumers validate the contract
# cheaply and log a meaningful error on mismatch.
uint32 observation_dim
# Unbounded array ([]) because the observation size may grow in later
# phases; float32 matches typical neural network input precision.
float32[] observation
