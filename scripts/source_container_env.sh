#!/usr/bin/env bash
# Source from Isaac ROS container terminals when ~/.bashrc is read-only (bind-mounted :ro).
#   source /workspaces/isaac_ros-dev/src/spark_isaac_mycobot_demo/scripts/source_container_env.sh
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
