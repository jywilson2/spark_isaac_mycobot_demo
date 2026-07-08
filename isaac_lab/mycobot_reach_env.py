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

"""Isaac Lab DirectRLEnv for Phase 2 EE reach-to-target PPO training.

Tutorial overview (see spec.md § Phase 2):
  1. Each episode samples a reachable 3D target; a red kinematic sphere marks it.
  2. Observations give the EE-to-target vector + joint proprioception (no vision).
  3. Actions are **joint position deltas** (Δq); the policy learns IK-style coordination.
  4. Rewards use potential-based distance reduction + time penalty + reach bonus.
  5. Curriculum widens target difficulty as rolling success improves.

Isaac Lab DirectRLEnv reference:
  https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.html
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
    EE_REACH_TOLERANCE_M,
    END_EFFECTOR_BODY_NAME,
    OBSERVATION_DIM,
    REVOLUTE_JOINT_NAMES,
    ReachTaskConfig,
    compute_reach_task_reward,
    episode_reach_success,
    resolve_curriculum_stage,
    sample_demo_workspace_xyz,
    sample_reachable_ee_xyz,
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

TASK_ID = 'Spark-MyCobot-Reach-Direct-v0'
EPISODE_METRICS_HISTORY = 128
TARGET_MARKER_RADIUS_M = 0.012

REACH_READY_JOINT_POS = {
    'joint2_to_joint1': 0.0,
    'joint3_to_joint2': 0.0,
    'joint4_to_joint3': 0.0,
    'joint5_to_joint4': 0.0,
    'joint6_to_joint5': 0.0,
    'joint6output_to_joint6': 0.0,
}


@configclass
class MyCobotReachEnvCfg(DirectRLEnvCfg):
    """Configuration for Phase 2 EE reach DirectRLEnv."""

    decimation = 2
    episode_length_s = 30.0
    action_space = ACTION_DIM
    observation_space = OBSERVATION_DIM
    state_space = 0
    action_scale: float = 0.12
    # EMA on raw joint deltas — smooth acceleration/deceleration (spec.md).
    action_smoothing_alpha: float = 0.35
    robot_usd_path: str = str(DEFAULT_ROBOT_USD)
    checkpoint_dir: str = str(DEFAULT_CHECKPOINT_DIR)
    # ``curriculum`` — staged easy→hard sampling during training.
    # ``workspace`` — uniform over the full reach envelope (play).
    # ``demo`` — stratified workspace cells + min separation (showcase).
    # ``precision`` — demo distribution + direct-path reward shaping (fine-tune).
    target_sampling: str = 'curriculum'
    reach_tolerance_m: float = EE_REACH_TOLERANCE_M
    direct_path_shaping: bool = False

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

    target_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path='/World/envs/env_.*/TargetMarker',
        spawn=sim_utils.SphereCfg(
            radius=TARGET_MARKER_RADIUS_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.05, 0.05)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, 0.0, 0.12)),
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


class MyCobotReachEnv(DirectRLEnv):
    """GPU-native Isaac Lab environment for efficient EE reach PPO (Phase 2)."""

    cfg: MyCobotReachEnvCfg

    def __init__(
        self,
        cfg: MyCobotReachEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        self._task_cfg = ReachTaskConfig(
            action_scale=cfg.action_scale,
            reach_tolerance_m=cfg.reach_tolerance_m,
            direct_path_shaping=cfg.direct_path_shaping,
        )
        self._ee_body_idx = 0
        self._target_ee_pos = None
        self._episode_reached = None
        self._episode_steps_to_reach = None
        self._reach_success_history: deque[bool] = deque(maxlen=EPISODE_METRICS_HISTORY)
        self._time_to_reach_history: deque[float] = deque(maxlen=EPISODE_METRICS_HISTORY)
        self._prev_distance: torch.Tensor | None = None
        self._prev_ee_pos: torch.Tensor | None = None
        self._prev_step_actions: torch.Tensor | None = None
        self._action_jerk: torch.Tensor | None = None
        self._curriculum_stage_name = 'near_ee'
        self._demo_target_initialized = False
        self._last_demo_targets: torch.Tensor | None = None
        self._terminal_episode_outcomes: dict[int, dict[str, Any]] = {}
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
        self._target_marker = RigidObject(self.cfg.target_marker)
        self._table = RigidObject(self.cfg.table)
        spawn_ground_plane(prim_path='/World/ground', cfg=GroundPlaneCfg())

        src, dest = '/World/envs/env_0', '/World/envs/env_{}'
        pos = cloner.grid_transforms(
            self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device)[0]
        plan = cloner.ClonePlan.from_env_0(src, dest, self.scene.num_envs, self.device, pos)
        cloner.replicate(plan, stage=self.scene.stage)
        if self.device == 'cpu':
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations['robot'] = self._robot
        self.scene.rigid_objects['target_marker'] = self._target_marker
        self.scene.rigid_objects['table'] = self._table

        light_cfg = sim_utils.DomeLightCfg(intensity=900.0)
        light_cfg.func('/World/DomeLight', light_cfg)

    def _build_reach_ready_tensor(self) -> torch.Tensor:
        joint_pos = [REACH_READY_JOINT_POS[name] for name in REVOLUTE_JOINT_NAMES]
        return torch.tensor(joint_pos, dtype=torch.float, device=self.device)

    def _rolling_reach_success_rate(self) -> float:
        if not self._reach_success_history:
            return 0.0
        return sum(self._reach_success_history) / len(self._reach_success_history)

    def _sample_target_positions(
        self,
        count: int,
        *,
        near_ee_centers: torch.Tensor | None = None,
        previous_targets: list[tuple[float, float, float]] | None = None,
    ) -> torch.Tensor:
        """Sample targets within the arm reach envelope (spec.md § Phase 2)."""

        if self.cfg.target_sampling == 'demo':
            self._curriculum_stage_name = 'demo'
            batch = sample_demo_workspace_xyz(
                count,
                previous_targets=previous_targets,
            )
            return torch.tensor(batch, dtype=torch.float, device=self.device)

        if self.cfg.target_sampling == 'precision':
            self._curriculum_stage_name = 'precision'
            batch = sample_demo_workspace_xyz(
                count,
                previous_targets=previous_targets,
            )
            return torch.tensor(batch, dtype=torch.float, device=self.device)

        if self.cfg.target_sampling == 'workspace':
            self._curriculum_stage_name = 'workspace'
            batch = sample_reachable_ee_xyz(count, easy_fraction=0.0)
            return torch.tensor(batch, dtype=torch.float, device=self.device)

        recent_rate = self._rolling_reach_success_rate()
        stage = resolve_curriculum_stage(recent_rate)
        self._curriculum_stage_name = stage.name
        samples: list[tuple[float, float, float]] = []
        for idx in range(count):
            near_center = None
            if near_ee_centers is not None and idx < near_ee_centers.shape[0]:
                center = near_ee_centers[idx]
                near_center = (
                    float(center[0].item()),
                    float(center[1].item()),
                    float(center[2].item()),
                )
            batch = sample_reachable_ee_xyz(
                1,
                easy_fraction=stage.easy_fraction,
                near_ee_center=near_center,
                near_ee_radius_m=stage.near_ee_radius_m,
            )
            samples.extend(batch)
        return torch.tensor(samples, dtype=torch.float, device=self.device)

    def _reset_robot_to_reach_ready(self, env_ids: torch.Tensor) -> None:
        joint_pos = self._reach_ready_joint_pos.unsqueeze(0).expand(env_ids.numel(), -1)
        joint_vel = torch.zeros_like(joint_pos)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)

    def _reset_target_marker(self, env_ids: torch.Tensor) -> None:
        ee_pos = self._ee_position_base()[env_ids]
        previous_targets = None
        if self.cfg.target_sampling in ('demo', 'precision') and self._demo_target_initialized:
            previous_targets = [
                tuple(self._last_demo_targets[i].detach().cpu().tolist())
                for i in env_ids.tolist()
            ]
        local_pos = self._sample_target_positions(
            env_ids.numel(),
            near_ee_centers=ee_pos,
            previous_targets=previous_targets,
        )
        self._target_ee_pos[env_ids] = local_pos
        if self.cfg.target_sampling in ('demo', 'precision'):
            if self._last_demo_targets is None:
                self._last_demo_targets = torch.zeros_like(self._target_ee_pos)
            self._last_demo_targets[env_ids] = local_pos
            self._demo_target_initialized = True
        world_pos = local_pos + self.scene.env_origins[env_ids]
        root_state = self._target_marker.data.default_root_state[env_ids].clone()
        root_state[:, 0:3] = world_pos
        root_state[:, 7:13] = 0.0
        self._target_marker.write_root_state_to_sim(root_state, env_ids=env_ids)

    def _settle_physics_after_reset(self) -> None:
        self.scene.write_data_to_sim()
        for _ in range(4):
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

    def _init_episode_state(self) -> None:
        self._target_ee_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self._episode_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._episode_steps_to_reach = torch.full(
            (self.num_envs,),
            float(self.max_episode_length),
            dtype=torch.float,
            device=self.device,
        )

    def _ee_position_base(self) -> torch.Tensor:
        return self._robot.data.body_pos_w.torch[:, self._ee_body_idx] - self.scene.env_origins

    def _distance_to_target(self, ee_pos: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(ee_pos - self._target_ee_pos, dim=-1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Bound raw actions, then EMA-smooth for gradual accel/decel (spec.md).
        raw = torch.clamp(actions, -1.0, 1.0)
        alpha = float(self.cfg.action_smoothing_alpha)
        prev = self._prev_step_actions
        if prev is None or prev.shape != raw.shape:
            smoothed = raw
            self._action_jerk = torch.zeros(raw.shape[0], device=raw.device)
        else:
            smoothed = alpha * raw + (1.0 - alpha) * prev
            self._action_jerk = (smoothed - prev).abs().mean(dim=-1)
        self._prev_step_actions = smoothed.detach().clone()
        self.actions = smoothed

    def _apply_action(self) -> None:
        """Apply learned joint deltas — the policy is the IK controller (spec.md)."""

        targets = self._robot.data.joint_pos.torch + self.cfg.action_scale * self.actions
        limits = getattr(self._robot.data, 'soft_joint_pos_limits', None)
        if limits is not None and limits.numel() > 0:
            lower = limits[..., 0]
            upper = limits[..., 1]
            if lower.shape == targets.shape:
                targets = torch.clamp(targets, lower, upper)
        self._robot.set_joint_position_target_index(target=targets)

    def _get_observations(self) -> dict:
        ee_pos = self._ee_position_base()
        ee_delta = self._target_ee_pos - ee_pos
        scale = max(self._task_cfg.max_reach_m, 1e-6)
        normalized_delta = ee_delta / scale
        target_valid = torch.ones((self.num_envs, 1), device=self.device)
        reached_flag = self._episode_reached.float().unsqueeze(1)
        joint_positions = self._robot.data.joint_pos.torch
        obs = torch.cat(
            [normalized_delta, target_valid, reached_flag, joint_positions],
            dim=-1,
        )
        obs = self._sanitize_observations(obs)
        return {'policy': obs}

    def _sanitize_observations(self, obs: torch.Tensor) -> torch.Tensor:
        """Reset any env whose state went non-finite instead of crashing the run.

        A rare PhysX solver blow-up can produce NaN body poses; rsl_rl aborts
        training on the first NaN observation. Resetting the offending envs
        costs one episode each and keeps the run alive.
        """

        bad_envs = (~torch.isfinite(obs)).any(dim=-1).nonzero(as_tuple=False).squeeze(-1)
        if bad_envs.numel() == 0:
            return obs
        print(
            f'[mycobot_reach_env] WARNING: non-finite observations in '
            f'{bad_envs.numel()} env(s) — resetting them.',
        )
        self._reset_idx(bad_envs)
        self.scene.write_data_to_sim()
        obs[bad_envs] = 0.0
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_rewards(self) -> torch.Tensor:
        ee_pos = self._ee_position_base()
        distance = self._distance_to_target(ee_pos)
        if self._prev_distance is None:
            self._prev_distance = distance.detach().clone()

        from isaac_lab.mdp_core import compute_lateral_step_m  # noqa: WPS433

        mean_abs_actions = self.actions.abs().mean(dim=-1)
        if self._action_jerk is not None and self._action_jerk.shape[0] == self.num_envs:
            mean_jerk = self._action_jerk
        else:
            mean_jerk = torch.zeros(self.num_envs, device=self.device)
        rewards = torch.zeros(self.num_envs, device=self.device)
        for env_idx in range(self.num_envs):
            lateral = 0.0
            if self._prev_ee_pos is not None:
                lateral = compute_lateral_step_m(
                    float(ee_pos[env_idx, 0].item() - self._prev_ee_pos[env_idx, 0].item()),
                    float(ee_pos[env_idx, 1].item() - self._prev_ee_pos[env_idx, 1].item()),
                    float(ee_pos[env_idx, 2].item() - self._prev_ee_pos[env_idx, 2].item()),
                    float(self._target_ee_pos[env_idx, 0].item() - ee_pos[env_idx, 0].item()),
                    float(self._target_ee_pos[env_idx, 1].item() - ee_pos[env_idx, 1].item()),
                    float(self._target_ee_pos[env_idx, 2].item() - ee_pos[env_idx, 2].item()),
                )
            reward, _dist = compute_reach_task_reward(
                end_effector_x=float(ee_pos[env_idx, 0].item()),
                end_effector_y=float(ee_pos[env_idx, 1].item()),
                end_effector_z=float(ee_pos[env_idx, 2].item()),
                target_x=float(self._target_ee_pos[env_idx, 0].item()),
                target_y=float(self._target_ee_pos[env_idx, 1].item()),
                target_z=float(self._target_ee_pos[env_idx, 2].item()),
                prev_distance_m=float(self._prev_distance[env_idx].item()),
                mean_abs_action=float(mean_abs_actions[env_idx].item()),
                mean_action_jerk=float(mean_jerk[env_idx].item()),
                lateral_step_m=lateral,
                cfg=self._task_cfg,
            )
            rewards[env_idx] = reward
        self._prev_distance = distance.detach().clone()
        self._prev_ee_pos = ee_pos.detach().clone()

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        timeout_penalty = self._task_cfg.timeout_penalty
        rewards = torch.where(
            time_out & ~self._episode_reached,
            rewards - timeout_penalty,
            rewards,
        )
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos = self._ee_position_base()
        distance = self._distance_to_target(ee_pos)
        reached_now = distance <= self._task_cfg.reach_tolerance_m
        newly_reached = reached_now & ~self._episode_reached
        if torch.any(newly_reached):
            self._episode_steps_to_reach[newly_reached] = self.episode_length_buf[newly_reached].float()
        self._episode_reached |= reached_now

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = self._episode_reached
        return terminated, time_out

    def _record_episode_outcomes(self, env_ids: torch.Tensor) -> None:
        step_dt = self.cfg.sim.dt * self.cfg.decimation
        ee_pos = self._ee_position_base()
        distance = self._distance_to_target(ee_pos)
        for env_idx in env_ids.detach().cpu().tolist():
            reached = bool(self._episode_reached[env_idx].item())
            steps = float(self._episode_steps_to_reach[env_idx].item())
            self._terminal_episode_outcomes[env_idx] = {
                'reached': reached,
                'time_to_reach_s': steps * step_dt if reached else None,
                'target_xyz': self._target_ee_pos[env_idx].detach().cpu().tolist(),
                'ee_xyz': ee_pos[env_idx].detach().cpu().tolist(),
                'distance_m': float(distance[env_idx].item()),
            }
            self._reach_success_history.append(episode_reach_success(reached))
            if reached:
                self._time_to_reach_history.append(steps * step_dt)
            else:
                self._time_to_reach_history.append(float(self.max_episode_length) * step_dt)

    def pop_episode_outcome(self, env_idx: int = 0) -> dict[str, Any] | None:
        """Return terminal episode metrics captured before Isaac Lab auto-reset."""

        return self._terminal_episode_outcomes.pop(env_idx, None)

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
            self._reset_target_marker(env_ids_tensor)
            self._settle_physics_after_reset()
            self._episode_reached[env_ids_tensor] = False
            self._episode_steps_to_reach[env_ids_tensor] = float(self.max_episode_length)
            self._prev_distance = None
            if self._prev_ee_pos is not None:
                settled_ee = self._ee_position_base()[env_ids_tensor]
                self._prev_ee_pos[env_ids_tensor] = settled_ee.detach()
            if self._prev_step_actions is not None:
                self._prev_step_actions[env_ids_tensor] = 0.0
            if self._action_jerk is not None:
                self._action_jerk[env_ids_tensor] = 0.0

    def get_task_metrics(self) -> dict[str, float]:
        if not self._reach_success_history:
            return {
                'reach_success_rate': 0.0,
                'mean_time_to_reach_s': 0.0,
                'target_reach_tolerance_m': self._task_cfg.reach_tolerance_m,
                'curriculum_stage': self._curriculum_stage_name,
            }
        reach_rate = sum(self._reach_success_history) / len(self._reach_success_history)
        mean_time = sum(self._time_to_reach_history) / len(self._time_to_reach_history)
        return {
            'reach_success_rate': float(reach_rate),
            'mean_time_to_reach_s': float(mean_time),
            'target_reach_tolerance_m': self._task_cfg.reach_tolerance_m,
            'curriculum_stage': self._curriculum_stage_name,
        }


def make_env_cfg(
    *,
    num_envs: int = 1,
    robot_usd_path: str | None = None,
    checkpoint_dir: str | None = None,
    seed: int | None = None,
    target_sampling: str = 'curriculum',
    episode_length_s: float | None = None,
    action_scale: float | None = None,
    reach_tolerance_m: float | None = None,
    direct_path_shaping: bool | None = None,
) -> MyCobotReachEnvCfg:
    cfg = MyCobotReachEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.target_sampling = target_sampling
    if episode_length_s is not None:
        cfg.episode_length_s = episode_length_s
    if action_scale is not None:
        cfg.action_scale = action_scale
    if reach_tolerance_m is not None:
        cfg.reach_tolerance_m = reach_tolerance_m
    if direct_path_shaping is not None:
        cfg.direct_path_shaping = direct_path_shaping
    usd_path = robot_usd_path or str(DEFAULT_ROBOT_USD)
    cfg.robot_usd_path = usd_path
    cfg.robot.spawn.usd_path = usd_path
    if checkpoint_dir is not None:
        cfg.checkpoint_dir = checkpoint_dir
    if seed is not None:
        cfg.seed = seed
    _configure_sim_physics(cfg)
    return cfg


def _configure_sim_physics(cfg: MyCobotReachEnvCfg) -> None:
    from isaaclab_physx.physics import PhysxCfg  # noqa: WPS433

    if cfg.sim.physics is None:
        cfg.sim.physics = PhysxCfg()
    cfg.sim.physics.enable_external_forces_every_iteration = True


def register_mycobot_env() -> None:
    if TASK_ID in gym.registry:
        return
    gym.register(
        id=TASK_ID,
        entry_point=f'{__name__}:MyCobotReachEnv',
        disable_env_checker=True,
        kwargs={'env_cfg_entry_point': f'{__name__}:MyCobotReachEnvCfg'},
    )


register_mycobot_env()
