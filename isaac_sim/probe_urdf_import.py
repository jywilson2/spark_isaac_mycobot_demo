#!/usr/bin/env python3
"""Minimal Isaac Sim URDF import probe for fast host-side iteration.

Run via:
    ./scripts/host/iter_urdf_import.sh

Exits 0 when a robot USD is produced; non-zero with traceback otherwise.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaac_sim.build_mycobot_limo_cobot_scene import prepare_robot_assets  # noqa: E402
from isaac_sim.urdf_import import import_urdf_to_usd  # noqa: E402
from isaac_sim.urdf_utils import default_upstream_urdf  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Probe Isaac Sim URDF import only.')
    parser.add_argument('--repo-root', type=Path, default=REPO_ROOT)
    parser.add_argument(
        '--keep-prepared',
        action='store_true',
        help='Reuse existing prepared URDF/meshes if present',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    upstream_urdf = default_upstream_urdf(repo_root)
    if not upstream_urdf.is_file():
        print(f'Missing upstream URDF: {upstream_urdf}', file=sys.stderr)
        return 1

    prepared_urdf = repo_root / 'assets' / 'robots' / 'mycobot_280_m5_limo_cobot' / 'mycobot_280_m5_limo_cobot.urdf'
    if args.keep_prepared and prepared_urdf.is_file():
        print(f'Reusing prepared URDF: {prepared_urdf}')
    else:
        prepared_urdf = prepare_robot_assets(repo_root, upstream_urdf)
        print(f'Prepared URDF: {prepared_urdf}')

    robot_usd = prepared_urdf.with_suffix('.usd')

    try:
        from isaacsim import SimulationApp  # noqa: WPS433

        simulation_app = SimulationApp({'headless': True})
        try:
            print('Importing URDF via isaacsim.asset.importer.urdf ...')
            output_usd = import_urdf_to_usd(
                prepared_urdf=prepared_urdf,
                output_usd=robot_usd,
                simulation_app=simulation_app,
            )
            if not output_usd.is_file() or output_usd.stat().st_size == 0:
                print(f'URDF import probe failed: empty or missing {output_usd}', file=sys.stderr)
                return 1
            print(f'Robot USD: {output_usd}')
            print(f'Size bytes: {output_usd.stat().st_size}')
        finally:
            simulation_app.close()
    except Exception as exc:  # noqa: BLE001
        print(f'URDF import probe failed: {exc}', file=sys.stderr)
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
