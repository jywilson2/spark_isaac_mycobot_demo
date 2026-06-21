#!/usr/bin/env python3
"""Run the MyCobot Limo Cobot live Isaac Sim scene with ROS 2 bridge enabled.

Host-side entry point (outside Docker):

    ${ISAACSIM_PATH}/python.sh \\
        /path/to/spark_isaac_mycobot_demo/isaac_sim/run_mycobot_live_sim.py

Optional flags:

    --scene-usd PATH    Load an existing scene USD (default: assets/scenes/...)
    --build-scene       Rebuild scene USD from URDF before loading
    --headless          Run without GUI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_USD = REPO_ROOT / 'assets' / 'scenes' / 'mycobot_280_m5_limo_cobot.usd'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run MyCobot live Isaac Sim with ROS 2 bridge.')
    parser.add_argument('--repo-root', type=Path, default=REPO_ROOT)
    parser.add_argument('--scene-usd', type=Path, default=DEFAULT_SCENE_USD)
    parser.add_argument('--build-scene', action='store_true', help='Rebuild USD from URDF first')
    parser.add_argument('--headless', action='store_true')
    return parser.parse_args()


def run_live_sim(args: argparse.Namespace) -> None:
    if args.build_scene or not args.scene_usd.is_file():
        from isaac_sim.build_mycobot_limo_cobot_scene import (  # noqa: WPS433
            build_scene,
            parse_args as parse_build_args,
        )

        build_args = parse_build_args()
        build_args.repo_root = args.repo_root
        build_args.save_usd = args.scene_usd
        build_args.headless = args.headless
        build_args.no_play = True
        build_scene(build_args)

    from isaacsim import SimulationApp  # noqa: WPS433

    simulation_app = SimulationApp({'headless': args.headless})

    import omni.timeline  # noqa: WPS433
    import omni.usd  # noqa: WPS433
    from isaacsim.core.utils.stage import open_stage  # noqa: WPS433

    from isaac_sim.ros2_bridge_config import (  # noqa: WPS433
        ROBOT_PRIM_PATH,
        configure_live_ros2_bridge,
    )

    scene_path = str(args.scene_usd.resolve())
    if not open_stage(scene_path):
        raise RuntimeError(f'Failed to open scene USD: {scene_path}')

    stage = omni.usd.get_context().get_stage()
    robot_prim = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    robot_path = ROBOT_PRIM_PATH if robot_prim.IsValid() else str(robot_prim.GetPath())

    configure_live_ros2_bridge(robot_prim_path=robot_path)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    print(f'Live sim running with scene: {scene_path}')
    print('ROS 2 topics:')
    print('  /clock')
    print('  /mycobot/joint_commands')
    print('  /mycobot/joint_states')
    print('  /mycobot/camera/rgb')
    print('Press Ctrl+C in the terminal to stop.')

    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass
    finally:
        timeline.stop()
        simulation_app.close()


def main() -> int:
    args = parse_args()
    try:
        run_live_sim(args)
    except ImportError as exc:
        print(
            'Isaac Sim Python modules are unavailable.\n'
            'Run with ${ISAACSIM_PATH}/python.sh on the Isaac Sim host.',
            file=sys.stderr,
        )
        print(f'ImportError: {exc}', file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f'Live sim failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
