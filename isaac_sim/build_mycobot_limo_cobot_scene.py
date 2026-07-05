#!/usr/bin/env python3
"""Build the MyCobot 280 M5 (Limo Cobot arm) Isaac Sim workspace scene.

Run on the Isaac Sim host (not inside the Isaac ROS Docker container):

    ${ISAACSIM_PATH}/python.sh \\
        /path/to/spark_isaac_mycobot_demo/isaac_sim/build_mycobot_limo_cobot_scene.py

Optional flags:

    --headless          Run without a GUI (still writes USD when --save-usd is set)
    --save-usd PATH     Output USD scene path (default: assets/scenes/...)
    --no-play           Do not auto-start timeline playback after load
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaac_sim.urdf_utils import (  # noqa: E402
    default_prepared_urdf,
    default_upstream_urdf,
    sanitize_collada_materials_in_dir,
    write_isaac_ready_urdf,
)
from isaac_sim.ros2_bridge_config import (  # noqa: E402
    ROBOT_PRIM_PATH,
    configure_live_ros2_bridge,
)
from isaac_sim.urdf_import import (  # noqa: E402
    URDF_IMPORTER_EXTENSION,
    add_robot_reference_to_stage,
    import_urdf_to_usd,
)

DEFAULT_SCENE_USD = (
    REPO_ROOT / "assets" / "scenes" / "mycobot_280_m5_limo_cobot.usd"
)
PREPARED_ROBOT_DIR = (
    REPO_ROOT / "assets" / "robots" / "mycobot_280_m5_limo_cobot"
)

REVOLUTE_JOINTS = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import the Limo Cobot myCobot 280 M5 URDF into Isaac Sim and "
            "assemble a pick-and-place workspace scene."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Path to spark_isaac_mycobot_demo (default: parent of isaac_sim/)",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Override upstream URDF path (default: third_party/mycobot_ros2 ...)",
    )
    parser.add_argument(
        "--save-usd",
        type=Path,
        default=DEFAULT_SCENE_USD,
        help=f"USD output path (default: {DEFAULT_SCENE_USD})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim without a GUI window",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Do not press Play after the scene is built",
    )
    parser.add_argument(
        "--with-ros2-bridge",
        action="store_true",
        help="Embed live ROS 2 OmniGraph bridge in the saved USD",
    )
    return parser.parse_args()


def prepare_robot_assets(repo_root: Path, upstream_urdf: Path) -> Path:
    """Copy meshes and write an Isaac-ready URDF beside them."""

    mesh_source_dir = upstream_urdf.parent
    prepared_urdf = default_prepared_urdf(repo_root)
    prepared_dir = prepared_urdf.parent

    if prepared_dir.exists():
        shutil.rmtree(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    for mesh_path in mesh_source_dir.iterdir():
        if mesh_path.is_file() and mesh_path.suffix.lower() in {".dae", ".png"}:
            shutil.copy2(mesh_path, prepared_dir / mesh_path.name)

    write_isaac_ready_urdf(
        upstream_urdf,
        prepared_urdf,
        mesh_dir=prepared_dir,
    )
    sanitize_collada_materials_in_dir(prepared_dir)
    return prepared_urdf


def build_scene(args: argparse.Namespace) -> Path:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            'headless': args.headless,
            'extra_args': [
                '--enable',
                URDF_IMPORTER_EXTENSION,
            ],
        }
    )

    import omni.timeline  # noqa: WPS433
    import omni.usd  # noqa: WPS433
    from isaacsim.core.utils.stage import create_new_stage  # noqa: WPS433
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: WPS433

    repo_root = args.repo_root.resolve()
    upstream_urdf = (args.urdf or default_upstream_urdf(repo_root)).resolve()
    if not upstream_urdf.is_file():
        raise FileNotFoundError(
            "Upstream URDF missing. Initialize submodules:\n"
            "  git submodule update --init --recursive"
        )

    prepared_urdf = prepare_robot_assets(repo_root, upstream_urdf)
    robot_usd = prepared_urdf.with_suffix('.usd')

    robot_usd_path = import_urdf_to_usd(
        prepared_urdf=prepared_urdf,
        output_usd=robot_usd,
        simulation_app=simulation_app,
    )

    create_new_stage()

    prim_path = add_robot_reference_to_stage(robot_usd_path, ROBOT_PRIM_PATH)

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    world_path = Sdf.Path("/World")
    world = UsdGeom.Xform.Define(stage, world_path)

    ground_path = world_path.AppendPath("GroundPlane")
    ground = UsdGeom.Cube.Define(stage, ground_path)
    ground.GetSizeAttr().Set(10.0)
    ground_xform = UsdGeom.Xformable(ground)
    ground_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.05))
    ground_xform.AddScaleOp().Set(Gf.Vec3d(10.0, 10.0, 0.01))

    table_path = world_path.AppendPath("WorkspaceTable")
    table = UsdGeom.Cube.Define(stage, table_path)
    table.GetSizeAttr().Set(1.0)
    table_xform = UsdGeom.Xformable(table)
    table_xform.AddTranslateOp().Set(Gf.Vec3d(0.35, 0.0, 0.375))
    table_xform.AddScaleOp().Set(Gf.Vec3d(0.6, 0.4, 0.75))

    block_path = world_path.AppendPath("PickBlock")
    block = UsdGeom.Cube.Define(stage, block_path)
    block.GetSizeAttr().Set(0.04)
    block_xform = UsdGeom.Xformable(block)
    block_xform.AddTranslateOp().Set(Gf.Vec3d(0.35, 0.0, 0.795))

    camera_path = world_path.AppendPath("WorkspaceCamera")
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(0.35, -0.55, 0.95))
    camera_xform.AddRotateXYZOp().Set(Gf.Vec3d(55.0, 0.0, 0.0))

    dome_path = world_path.AppendPath("DomeLight")
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.CreateIntensityAttr().Set(900.0)

    robot_prim = stage.GetPrimAtPath(prim_path)
    if robot_prim.IsValid():
        robot_xform = UsdGeom.Xformable(robot_prim)
        robot_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.78))

    block_material = UsdShade.Material.Define(stage, world_path.AppendPath("PickBlockMaterial"))
    shader = UsdShade.Shader.Define(stage, block_material.GetPath().AppendPath("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.1, 0.1))
    block_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(block).Bind(block_material)

    if args.with_ros2_bridge:
        try:
            configure_live_ros2_bridge(
                robot_prim_path=str(prim_path),
                camera_prim_path=str(camera_path),
            )
        except Exception as exc:  # noqa: BLE001 - bridge can be added at live sim runtime
            print(
                f'Warning: could not embed ROS 2 bridge in scene USD ({exc}). '
                'run_live_sim.sh configures the bridge when loading the scene.',
                file=sys.stderr,
            )

    save_path = args.save_usd.resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(save_path))

    print(f"Imported robot prim: {prim_path}")
    print(f"Revolute joints: {', '.join(REVOLUTE_JOINTS)}")
    print(f"Prepared URDF: {prepared_urdf}")
    print(f"Saved scene USD: {save_path}")

    if not args.no_play:
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

    for _ in range(5):
        simulation_app.update()

    simulation_app.close()
    return save_path


def main() -> int:
    args = parse_args()
    try:
        build_scene(args)
    except ImportError as exc:
        print(
            "Isaac Sim Python modules are unavailable in this environment.\n"
            "Run this script with ${ISAACSIM_PATH}/python.sh on the Isaac Sim host.",
            file=sys.stderr,
        )
        print(f"ImportError: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface build failures to CLI
        import traceback

        print(f'Scene build failed: {exc}', file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
