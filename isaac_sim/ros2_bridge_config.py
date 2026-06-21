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

"""Live Isaac Sim ROS 2 topic names and OmniGraph bridge configuration."""

from __future__ import annotations

GRAPH_PATH = '/ActionGraph'
ROBOT_PRIM_PATH = '/World/mycobot_280_m5_limo_cobot'
CAMERA_PRIM_PATH = '/World/WorkspaceCamera'

JOINT_COMMANDS_TOPIC = '/mycobot/joint_commands'
JOINT_STATES_TOPIC = '/mycobot/joint_states'
CAMERA_RGB_TOPIC = '/mycobot/camera/rgb'
CAMERA_FRAME_ID = 'workspace_camera'


def enable_ros2_bridge_extensions() -> None:
    import omni.kit.app  # noqa: WPS433

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    for extension in (
        'isaacsim.ros2.bridge',
        'isaacsim.ros2.sim_control',
    ):
        extension_manager.set_extension_enabled_immediate(extension, True)


def configure_live_ros2_bridge(
    robot_prim_path: str = ROBOT_PRIM_PATH,
    camera_prim_path: str = CAMERA_PRIM_PATH,
    graph_path: str = GRAPH_PATH,
) -> None:
    """Create OmniGraph nodes for live joint and camera ROS 2 topics."""

    import omni.graph.core as og  # noqa: WPS433

    enable_ros2_bridge_extensions()

    og.Controller.edit(
        {'graph_path': graph_path, 'evaluator_name': 'execution'},
        {
            og.Controller.Keys.CREATE_NODES: [
                ('OnPlaybackTick', 'omni.graph.action.OnPlaybackTick'),
                ('ReadSimTime', 'isaacsim.core.nodes.IsaacReadSimulationTime'),
                ('PublishJointState', 'isaacsim.ros2.bridge.ROS2PublishJointState'),
                ('SubscribeJointState', 'isaacsim.ros2.bridge.ROS2SubscribeJointState'),
                ('ArticulationController', 'isaacsim.core.nodes.IsaacArticulationController'),
                ('PublishClock', 'isaacsim.ros2.bridge.ROS2PublishClock'),
                ('RenderProduct', 'isaacsim.core.nodes.IsaacCreateRenderProduct'),
                ('CameraHelper', 'isaacsim.ros2.bridge.ROS2CameraHelper'),
            ],
            og.Controller.Keys.CONNECT: [
                ('OnPlaybackTick.outputs:tick', 'PublishJointState.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'SubscribeJointState.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'ArticulationController.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'PublishClock.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'RenderProduct.inputs:execIn'),
                ('OnPlaybackTick.outputs:tick', 'CameraHelper.inputs:execIn'),
                ('ReadSimTime.outputs:simulationTime', 'PublishJointState.inputs:timeStamp'),
                ('SubscribeJointState.outputs:jointNames', 'ArticulationController.inputs:jointNames'),
                (
                    'SubscribeJointState.outputs:positionCommand',
                    'ArticulationController.inputs:positionCommand',
                ),
                (
                    'SubscribeJointState.outputs:velocityCommand',
                    'ArticulationController.inputs:velocityCommand',
                ),
                (
                    'SubscribeJointState.outputs:effortCommand',
                    'ArticulationController.inputs:effortCommand',
                ),
                (
                    'RenderProduct.outputs:renderProductPath',
                    'CameraHelper.inputs:renderProductPath',
                ),
            ],
            og.Controller.Keys.SET_VALUES: [
                ('ArticulationController.inputs:robotPath', robot_prim_path),
                ('PublishJointState.inputs:targetPrim', robot_prim_path),
                ('PublishJointState.inputs:topicName', JOINT_STATES_TOPIC),
                ('SubscribeJointState.inputs:topicName', JOINT_COMMANDS_TOPIC),
                ('PublishClock.inputs:topicName', '/clock'),
                ('RenderProduct.inputs:cameraPrim', camera_prim_path),
                ('RenderProduct.inputs:width', 640),
                ('RenderProduct.inputs:height', 480),
                ('CameraHelper.inputs:topicName', CAMERA_RGB_TOPIC),
                ('CameraHelper.inputs:frameId', CAMERA_FRAME_ID),
                ('CameraHelper.inputs:type', 'rgb'),
            ],
        },
    )
