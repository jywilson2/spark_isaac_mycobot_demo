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

"""Isaac Lab DirectRLEnv for MyCobot pick-and-place PPO training."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, UsdFileCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from isaac_lab.mdp_core import (
    ACTION_DIM,
    OBSERVATION_DIM,
    REVOLUTE_JOINT_NAMES,
    MdpTaskConfig,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_USD = (
    REPO_ROOT
    / 'assets'
    / 'robots'
    / 'mycobot_280_m5_limo_cobot'
    / 'mycobot_280_m5_limo_cobot'
    / 'mycobot_280_m5_limo_cobot.usda'
)
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / 'assets' / 'checkpoints' / 'isaac_lab_ppo'

TASK_ID = 'Spark-MyCobot-PickPlace-Direct-v0'


@configclass
class MyCobotPickPlaceEnvCfg(DirectRLEnvCfg):
    """Configuration for the MyCobot pick-and-place DirectRLEnv."""

    decimation = 2
    episode_length_s = 8.0
    action_space = ACTION_DIM
    observation_space = OBSERVATION_DIM
    state_space = 0
    action_scale: float = 0.05
    robot_usd_path: str = str(DEFAULT_ROBOT_USD)
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT_DIR)

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path='/World/envs/env_.*/Robot',
        spawn=UsdFileCfg(
            usd_path=str(DEFAULT_ROBOT_USD),
            activate_contact_sensors=False,
        ),
        actuators={
            'arm': ImplicitActuatorCfg(
                joint_names_expr=list(REVOLUTE_JOINT_NAMES),
                stiffness=80.0,
                damping=8.0,
            ),
        },
    )

    block: RigidObjectCfg = RigidObjectCfg(
        prim_path='/World/envs/env_.*/PickBlock',
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.82)),
    )


class MyCobotPickPlaceEnv(DirectRLEnv):
    """GPU-native Isaac Lab environment for MyCobot pick-and-place PPO."""

    cfg: MyCobotPickPlaceEnvCfg

    def __init__(
        self,
        cfg: MyCobotPickPlaceEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        self._task_cfg = MdpTaskConfig(action_scale=cfg.action_scale)
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self._block = RigidObject(self.cfg.block)
        spawn_ground_plane(prim_path='/World/ground', cfg=GroundPlaneCfg())

        src, dest = '/World/envs/env_0', '/World/envs/env_{}'
        pos = cloner.grid_transforms(
            self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.ClonePlan.from_env_0(src, dest, self.scene.num_envs, self.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)
        if self.device == 'cpu':
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations['robot'] = self._robot
        self.scene.rigid_objects['block'] = self._block

        light_cfg = sim_utils.DomeLightCfg(intensity=900.0)
        light_cfg.func('/World/DomeLight', light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        targets = self._robot.data.joint_pos.torch + self.cfg.action_scale * self.actions
        self._robot.set_joint_position_target_index(target=targets)

    def _get_observations(self) -> dict:
        block_pos = self._block.data.root_pos_w.torch - self.scene.env_origins
        centroid_x = torch.clamp((block_pos[:, 0] - 0.20) / 0.30, 0.0, 1.0)
        centroid_y = torch.clamp((block_pos[:, 1] + 0.20) / 0.40, 0.0, 1.0)
        ee_z = self._robot.data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2] + 0.12
        joint_positions = self._robot.data.joint_pos.torch

        obs = torch.cat(
            [
                centroid_x.unsqueeze(1),
                centroid_y.unsqueeze(1),
                (centroid_x - 0.05).unsqueeze(1),
                (centroid_y - 0.05).unsqueeze(1),
                (centroid_x + 0.05).unsqueeze(1),
                (centroid_y + 0.05).unsqueeze(1),
                ee_z.unsqueeze(1),
                torch.zeros((self.num_envs, 1), device=self.device),
                joint_positions,
            ],
            dim=-1,
        )
        return {'policy': obs}

    def _get_rewards(self) -> torch.Tensor:
        block_pos = self._block.data.root_pos_w.torch - self.scene.env_origins
        centroid_x = torch.clamp((block_pos[:, 0] - 0.20) / 0.30, 0.0, 1.0)
        centroid_y = torch.clamp((block_pos[:, 1] + 0.20) / 0.40, 0.0, 1.0)
        ee_pos = self._robot.data.root_pos_w.torch - self.scene.env_origins

        target_x = self._task_cfg.target_centroid_x
        target_y = self._task_cfg.target_centroid_y
        tracking = torch.clamp(
            1.0 - torch.sqrt((centroid_x - target_x) ** 2 + (centroid_y - target_y) ** 2),
            min=0.0,
        )
        alignment = torch.clamp(
            1.0 - torch.sqrt(
                (centroid_x - ee_pos[:, 0]) ** 2 + (centroid_y - ee_pos[:, 1]) ** 2),
            min=0.0,
        )
        lift = torch.clamp((block_pos[:, 2] - 0.78) / self._task_cfg.target_lift_height, 0.0, 1.0)
        return tracking + alignment + lift

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)


def make_env_cfg(
    *,
    num_envs: int = 1,
    robot_usd_path: str | None = None,
    checkpoint_dir: str | None = None,
) -> MyCobotPickPlaceEnvCfg:
    cfg = MyCobotPickPlaceEnvCfg()
    cfg.scene.num_envs = num_envs
    usd_path = robot_usd_path or str(DEFAULT_ROBOT_USD)
    cfg.robot_usd_path = usd_path
    cfg.robot.spawn.usd_path = usd_path
    if checkpoint_dir is not None:
        cfg.checkpoint_dir = checkpoint_dir
    return cfg


def register_mycobot_env() -> None:
    if TASK_ID in gym.registry:
        return
    gym.register(
        id=TASK_ID,
        entry_point=f'{__name__}:MyCobotPickPlaceEnv',
        disable_env_checker=True,
        kwargs={'env_cfg_entry_point': f'{__name__}:MyCobotPickPlaceEnvCfg'},
    )


register_mycobot_env()
