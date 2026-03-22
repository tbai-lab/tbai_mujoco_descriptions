#!/usr/bin/env python3
"""Spawn a robot in MuJoCo and simulate it interactively."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import tyro

ROBOTS_DIR = Path(__file__).parent / "robots"
ROBOT_CHOICES = sorted(
    d.name for d in ROBOTS_DIR.iterdir() if d.is_dir() and list(d.glob("scene.xml"))
)


@dataclass
class Args:
    """Spawn a robot in its scene and simulate interactively in the MuJoCo viewer."""

    robot: str | None = None
    """Robot to visualize. If not specified, lists available robots."""


def main() -> None:
    args = tyro.cli(Args)

    if args.robot is None:
        print("Available robots:")
        for name in ROBOT_CHOICES:
            print(f"  {name}")
        print(f"\nUsage: uv run view_robot.py --robot <name>")
        sys.exit(0)

    if args.robot not in ROBOT_CHOICES:
        print(f"Unknown robot: {args.robot}")
        print("Available robots:")
        for name in ROBOT_CHOICES:
            print(f"  {name}")
        sys.exit(1)

    scene_path = ROBOTS_DIR / args.robot / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
