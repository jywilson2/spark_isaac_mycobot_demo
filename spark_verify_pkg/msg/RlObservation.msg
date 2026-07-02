# Flattened RL observation vector (Phase 2), the policy network's input.
# Layout contract (defined in reward_function.build_observation_vector):
#   [0-1] block centroid, [2-5] bbox, [6] ee height, [7] grasp flag,
#   [8-13] six joint positions (rad). Total dimension: 14.
std_msgs/Header header
# Redundant with len(observation) but lets consumers validate the contract
# cheaply and log a meaningful error on mismatch.
uint32 observation_dim
# Unbounded array ([]) because the observation size may grow in later
# phases; float32 matches typical neural network input precision.
float32[] observation
