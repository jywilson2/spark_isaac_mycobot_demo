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

"""Phase 5 Isaac Lab env: red-block vision localization and contact-and-push."""

from __future__ import annotations

from collections import deque
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
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass

from isaac_lab.mdp_core import (
    ACTION_DIM,
    BLOCK_SPAWN_Z,
    END_EFFECTOR_BODY_NAME,
    OBSERVATION_DIM,
    REVOLUTE_JOINT_NAMES,
    MdpTaskConfig,
    compute_contact_push_reward,
    compute_motion_approach_reward,
    episode_push_task_success,
    sample_reachable_block_xy,
)
from isaac_lab.phase5_red_block.block_localization import localize_block_from_camera_batch
from isaac_lab.phase5_red_block.init_scan import (
    INIT_SCAN_STEPS_PER_WAYPOINT,
    build_waypoint_tensor,
    waypoint_count,
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

PHASE5_TASK_ID = 'Spark-MyCobot-RedBlock-Direct-v0'
EPISODE_METRICS_HISTORY = 128

# Near-zero joints match the USD spawn pose that already points the EE toward +x workspace.
REACH_READY_JOINT_POS = {
    'joint2_to_joint1': 0.0,
    'joint3_to_joint2': 0.0,
    'joint4_to_joint3': 0.0,
    'joint5_to_joint4': 0.0,
    'joint6_to_joint5': 0.0,
    'joint6output_to_joint6': 0.0,
}

EE_CAMERA_HEIGHT = 128
EE_CAMERA_WIDTH = 128


@configclass
class MyCobotRedBlockEnvCfg(DirectRLEnvCfg):
    """Configuration for Phase 5 red-block contact-and-push DirectRLEnv."""

    decimation = 2
    episode_length_s = 12.0
    action_space = ACTION_DIM
    observation_space = OBSERVATION_DIM
    state_space = 0
    action_scale: float = 0.12
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
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=REACH_READY_JOINT_POS,
        ),
        actuators={
            'arm': ImplicitActuatorCfg(
                joint_names_expr=list(REVOLUTE_JOINT_NAMES),
                stiffness=200.0,
                damping=20.0,
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, 0.0, BLOCK_SPAWN_Z)),
    )

    ee_camera: CameraCfg = CameraCfg(
        prim_path='/World/envs/env_.*/Robot/ee_camera',
        update_period=0.0,
        height=EE_CAMERA_HEIGHT,
        width=EE_CAMERA_WIDTH,
        data_types=['rgb', 'distance_to_camera'],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention='ros',
        ),
    )

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path='/World/envs/env_.*/WorkspaceTable',
        spawn=sim_utils.CuboidCfg(
            size=(0.60, 0.60, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.45)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.01)),
    )


class MyCobotRedBlockEnv(DirectRLEnv):
    """Phase 5 GPU env: red-block vision localization and contact-and-push."""

    cfg: MyCobotRedBlockEnvCfg

    def __init__(
        self,
        cfg: MyCobotRedBlockEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        self._task_cfg = MdpTaskConfig(action_scale=cfg.action_scale)
        self._ee_body_idx = 0
        self._episode_had_contact = None
        self._episode_max_push_m = None
        self._contact_block_xyz = None
        self._last_camera_detected = None
        self._target_ee_pos = None
        self._target_valid = None
        self._block_pose_base = None
        self._push_success_history: deque[bool] = deque(maxlen=EPISODE_METRICS_HISTORY)
        self._contact_history: deque[bool] = deque(maxlen=EPISODE_METRICS_HISTORY)
        self._push_distance_history: deque[float] = deque(maxlen=EPISODE_METRICS_HISTORY)
        self._prev_horizontal_dist: torch.Tensor | None = None
        self._prev_ee_z: torch.Tensor | None = None
        self._reach_ready_joint_pos: torch.Tensor | None = None
        super().__init__(cfg, render_mode, **kwargs)
        self._reach_ready_joint_pos = self._build_reach_ready_tensor()
        ee_indices, _ee_names = self._robot.find_bodies(END_EFFECTOR_BODY_NAME)
        if not ee_indices:
            available = ', '.join(self._robot.body_names[:12])
            raise RuntimeError(
                f'End-effector body {END_EFFECTOR_BODY_NAME!r} not found. '
                f'Available bodies include: {available}')
        self._ee_body_idx = ee_indices[0]
        self._init_episode_state()

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self._block = RigidObject(self.cfg.block)
        self._table = RigidObject(self.cfg.table)
        self._ee_camera = Camera(self.cfg.ee_camera)
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
        self.scene.rigid_objects['table'] = self._table
        self.scene.sensors['ee_camera'] = self._ee_camera

        light_cfg = sim_utils.DomeLightCfg(intensity=900.0)
        light_cfg.func('/World/DomeLight', light_cfg)

    def _build_reach_ready_tensor(self) -> torch.Tensor:
        joint_pos = []
        for name in REVOLUTE_JOINT_NAMES:
            joint_pos.append(REACH_READY_JOINT_POS[name])
        return torch.tensor(joint_pos, dtype=torch.float, device=self.device)

    def _sample_block_positions(self, count: int) -> torch.Tensor:
        samples = sample_reachable_block_xy(count)
        positions = torch.zeros((count, 3), device=self.device)
        for idx, (x, y) in enumerate(samples):
            positions[idx, 0] = x
            positions[idx, 1] = y
            positions[idx, 2] = BLOCK_SPAWN_Z
        return positions

    def _reset_robot_to_reach_ready(self, env_ids: torch.Tensor) -> None:
        joint_pos = self._reach_ready_joint_pos.unsqueeze(0).expand(env_ids.numel(), -1)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)

    def _reset_block_positions(self, env_ids: torch.Tensor) -> None:
        local_pos = self._sample_block_positions(env_ids.numel())
        world_pos = local_pos + self.scene.env_origins[env_ids]
        root_state = self._block.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = world_pos
        root_state[:, 7:13] = 0.0
        self._block.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _settle_physics_after_reset(self) -> None:
        self.scene.write_data_to_sim()
        for _ in range(8):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

    def _init_episode_state(self) -> None:
        self._episode_had_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_max_push_m = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._contact_block_xyz = torch.full(
            (self.num_envs, 3), float('nan'), dtype=torch.float, device=self.device)
        self._last_camera_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._target_ee_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self._target_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._block_pose_base = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

    def _world_positions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_pos = self._block.data.root_pos_w.torch - self.scene.env_origins
        robot_base = self._robot.data.root_pos_w.torch - self.scene.env_origins
        ee_pos = self._robot.data.body_pos_w.torch[:, self._ee_body_idx] - self.scene.env_origins
        return block_pos, robot_base, ee_pos

    def _contact_mask(
        self,
        block_pos: torch.Tensor,
        ee_pos: torch.Tensor,
    ) -> torch.Tensor:
        block_top_z = block_pos[:, 2] + self._task_cfg.block_half_size_m
        horizontal_dist = torch.linalg.vector_norm(block_pos[:, :2] - ee_pos[:, :2], dim=-1)
        vertical_gap = ee_pos[:, 2] - block_top_z
        return (
            (horizontal_dist <= self._task_cfg.contact_horizontal_tolerance_m)
            & (vertical_gap <= self._task_cfg.contact_vertical_max_gap_m)
            & (vertical_gap >= self._task_cfg.contact_vertical_min_gap_m)
        )

    def _update_push_state(
        self,
        block_pos: torch.Tensor,
        contact_mask: torch.Tensor,
    ) -> torch.Tensor:
        first_contact = contact_mask & ~self._episode_had_contact
        if torch.any(first_contact):
            self._contact_block_xyz[first_contact] = block_pos[first_contact]
        self._episode_had_contact |= contact_mask

        active = self._episode_had_contact & ~torch.isnan(self._contact_block_xyz[:, 0])
        push_distance = torch.zeros(self.num_envs, device=self.device)
        if torch.any(active):
            delta = block_pos[active] - self._contact_block_xyz[active]
            push_distance[active] = torch.linalg.vector_norm(delta, dim=-1)
        self._episode_max_push_m = torch.maximum(self._episode_max_push_m, push_distance)
        return push_distance

    def _sync_ee_camera_poses(self) -> None:
        ee_pos = self._robot.data.body_pos_w.torch[:, self._ee_body_idx]
        ee_quat = self._robot.data.body_quat_w.torch[:, self._ee_body_idx]
        self._ee_camera.set_world_poses(ee_pos, ee_quat, convention='ros')

    def _camera_pose_in_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        cam_pos = self._robot.data.body_pos_w.torch[:, self._ee_body_idx] - self.scene.env_origins
        cam_quat = self._robot.data.body_quat_w.torch[:, self._ee_body_idx]
        return cam_pos, cam_quat

    def _localize_from_camera(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = self._ee_camera.data.output['rgb']
        depth = self._ee_camera.data.output['distance_to_camera']
        cam_pos, cam_quat = self._camera_pose_in_base()
        block_pos, target_ee, detected, pose_valid = localize_block_from_camera_batch(
            rgb,
            depth,
            cam_pos,
            cam_quat,
            block_half_size_m=self._task_cfg.block_half_size_m,
            approach_offset_m=self._task_cfg.approach_offset_m,
        )
        self._last_camera_detected = detected
        return block_pos, target_ee, detected, pose_valid

    def _run_init_scan(self, env_ids: torch.Tensor) -> None:
        """Sweep predetermined joint poses until vision locks a target EE pose."""

        if env_ids.numel() == 0:
            return
        self._target_valid[env_ids] = False
        for waypoint_idx in range(waypoint_count()):
            waypoint = build_waypoint_tensor(
                waypoint_idx,
                device=self.device,
                dtype=self._robot.data.joint_pos.dtype,
            ).expand(env_ids.numel(), -1)
            self._robot.set_joint_position_target_index(target=waypoint, env_ids=env_ids)
            self.scene.write_data_to_sim()
            for _ in range(INIT_SCAN_STEPS_PER_WAYPOINT):
                self.sim.step(render=True)
                self.scene.update(dt=self.physics_dt)
                self._sync_ee_camera_poses()
                block_pos, target_ee, _detected, pose_valid = self._localize_from_camera()
                newly_valid = pose_valid & ~self._target_valid
                if torch.any(newly_valid):
                    self._target_valid[newly_valid] = True
                    self._target_ee_pos[newly_valid] = target_ee[newly_valid]
                    self._block_pose_base[newly_valid] = block_pos[newly_valid]
            if bool(torch.all(self._target_valid[env_ids]).item()):
                break

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._sync_ee_camera_poses()

    def _apply_action(self) -> None:
        targets = self._robot.data.joint_pos.torch + self.cfg.action_scale * self.actions
        self._robot.set_joint_position_target_index(target=targets)

    def _get_observations(self) -> dict:
        _block_pos, _robot_base, ee_pos = self._world_positions()
        ee_delta = self._target_ee_pos - ee_pos
        scale = max(self._task_cfg.max_reach_m, 1e-6)
        normalized_delta = ee_delta / scale
        target_valid = self._target_valid.float().unsqueeze(1)
        contact_flag = self._episode_had_contact.float().unsqueeze(1)
        joint_positions = self._robot.data.joint_pos.torch
        obs = torch.cat(
            [normalized_delta, target_valid, contact_flag, joint_positions],
            dim=-1,
        )
        return {'policy': obs}

    def _get_rewards(self) -> torch.Tensor:
        block_pos, robot_base, ee_pos = self._world_positions()
        contact_mask = self._contact_mask(block_pos, ee_pos)
        new_contact = contact_mask & ~self._episode_had_contact
        push_distance = self._update_push_state(block_pos, contact_mask)

        horizontal_dist = torch.linalg.vector_norm(block_pos[:, :2] - ee_pos[:, :2], dim=-1)
        distance_reward = torch.clamp(1.0 - horizontal_dist / 0.20, min=0.0)

        if self._prev_horizontal_dist is None:
            self._prev_horizontal_dist = horizontal_dist.detach().clone()
        approach_progress = torch.clamp(self._prev_horizontal_dist - horizontal_dist, min=0.0) * 8.0
        self._prev_horizontal_dist = horizontal_dist.detach().clone()

        block_top_z = block_pos[:, 2] + self._task_cfg.block_half_size_m
        vertical_gap = ee_pos[:, 2] - block_top_z
        close_xy = horizontal_dist < 0.12
        close_descend_bonus = close_xy.float() * torch.clamp(1.0 - vertical_gap / 0.25, min=0.0) * 2.0
        if self._prev_ee_z is None:
            self._prev_ee_z = ee_pos[:, 2].detach().clone()
        ee_descend = torch.clamp(self._prev_ee_z - ee_pos[:, 2], min=0.0)
        descend_step_bonus = close_xy.float() * ee_descend * 25.0
        self._prev_ee_z = ee_pos[:, 2].detach().clone()

        rewards = torch.zeros(self.num_envs, device=self.device)
        for env_idx in range(self.num_envs):
            target_valid = bool(self._target_valid[env_idx].item())
            target_ee = self._target_ee_pos[env_idx]
            motion_reward = compute_motion_approach_reward(
                end_effector_x=float(ee_pos[env_idx, 0].item()),
                end_effector_y=float(ee_pos[env_idx, 1].item()),
                end_effector_z=float(ee_pos[env_idx, 2].item()),
                target_ee_x=float(target_ee[0].item()),
                target_ee_y=float(target_ee[1].item()),
                target_ee_z=float(target_ee[2].item()),
                target_valid=target_valid,
            )
            rewards[env_idx] = compute_contact_push_reward(
                block_x=float(block_pos[env_idx, 0].item()),
                block_y=float(block_pos[env_idx, 1].item()),
                block_z=float(block_pos[env_idx, 2].item()),
                end_effector_x=float(ee_pos[env_idx, 0].item()),
                end_effector_y=float(ee_pos[env_idx, 1].item()),
                end_effector_z=float(ee_pos[env_idx, 2].item()),
                robot_base_x=float(robot_base[env_idx, 0].item()),
                robot_base_y=float(robot_base[env_idx, 1].item()),
                push_distance_m=float(push_distance[env_idx].item()),
                cfg=self._task_cfg,
            ) + motion_reward + 0.75 * float(distance_reward[env_idx].item()) + float(
                approach_progress[env_idx].item()
            )
            rewards[env_idx] += float(close_descend_bonus[env_idx].item()) + float(descend_step_bonus[env_idx].item())
            if bool(new_contact[env_idx].item()):
                rewards[env_idx] += 3.0
            rewards[env_idx] += float(push_distance[env_idx].item()) * 80.0
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        success = self._episode_max_push_m >= self._task_cfg.target_push_distance_m
        terminated = success
        return terminated, time_out

    def _record_episode_outcomes(self, env_ids: torch.Tensor) -> None:
        for env_idx in env_ids.detach().cpu().tolist():
            had_contact = bool(self._episode_had_contact[env_idx].item())
            max_push = float(self._episode_max_push_m[env_idx].item())
            self._contact_history.append(had_contact)
            self._push_distance_history.append(max_push)
            self._push_success_history.append(
                episode_push_task_success(
                    max_push,
                    target_push_distance_m=self._task_cfg.target_push_distance_m,
                    had_contact=had_contact,
                )
            )

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids_tensor = self._robot._ALL_INDICES
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.tensor(list(env_ids), device=self.device, dtype=torch.long)

        if env_ids_tensor.numel() > 0:
            self._record_episode_outcomes(env_ids_tensor)
        super()._reset_idx(env_ids)
        if env_ids_tensor.numel() > 0:
            self._reset_robot_to_reach_ready(env_ids_tensor)
            self._reset_block_positions(env_ids_tensor)
            self._settle_physics_after_reset()
            self._sync_ee_camera_poses()
            self._target_valid[env_ids_tensor] = False
            self._target_ee_pos[env_ids_tensor] = 0.0
            self._block_pose_base[env_ids_tensor] = 0.0
            self._run_init_scan(env_ids_tensor)
            self._episode_had_contact[env_ids_tensor] = False
            self._episode_max_push_m[env_ids_tensor] = 0.0
            self._contact_block_xyz[env_ids_tensor] = float('nan')
            self._prev_horizontal_dist = None
            self._prev_ee_z = None

    def get_task_metrics(self) -> dict[str, float]:
        if not self._push_success_history:
            return {
                'contact_rate': 0.0,
                'push_success_rate': 0.0,
                'mean_push_distance_m': 0.0,
                'target_push_distance_m': self._task_cfg.target_push_distance_m,
            }
        contact_rate = sum(self._contact_history) / len(self._contact_history)
        push_success_rate = sum(self._push_success_history) / len(self._push_success_history)
        mean_push = sum(self._push_distance_history) / len(self._push_distance_history)
        return {
            'contact_rate': float(contact_rate),
            'push_success_rate': float(push_success_rate),
            'mean_push_distance_m': float(mean_push),
            'target_push_distance_m': self._task_cfg.target_push_distance_m,
        }


def make_env_cfg(
    *,
    num_envs: int = 1,
    robot_usd_path: str | None = None,
    checkpoint_dir: str | None = None,
    seed: int | None = None,
) -> MyCobotRedBlockEnvCfg:
    cfg = MyCobotRedBlockEnvCfg()
    cfg.scene.num_envs = num_envs
    usd_path = robot_usd_path or str(DEFAULT_ROBOT_USD)
    cfg.robot_usd_path = usd_path
    cfg.robot.spawn.usd_path = usd_path
    if checkpoint_dir is not None:
        cfg.checkpoint_dir = checkpoint_dir
    if seed is not None:
        cfg.seed = seed
    _configure_sim_physics(cfg)
    return cfg


def _configure_sim_physics(cfg: MyCobotRedBlockEnvCfg) -> None:
    from isaaclab_physx.physics import PhysxCfg  # noqa: WPS433

    if cfg.sim.physics is None:
        cfg.sim.physics = PhysxCfg()
    cfg.sim.physics.enable_external_forces_every_iteration = True


def register_red_block_env() -> None:
    if PHASE5_TASK_ID in gym.registry:
        return
    gym.register(
        id=PHASE5_TASK_ID,
        entry_point='isaac_lab.phase5_red_block.red_block_env:MyCobotRedBlockEnv',
        disable_env_checker=True,
        kwargs={'env_cfg_entry_point': 'isaac_lab.phase5_red_block.red_block_env:MyCobotRedBlockEnvCfg'},
    )


register_red_block_env()
