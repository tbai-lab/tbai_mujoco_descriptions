#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro

ROBOTS_DIR = Path(__file__).parent / "robots"
ROBOT_CHOICES = sorted(
    d.name for d in ROBOTS_DIR.iterdir() if d.is_dir() and list(d.glob("scene.xml"))
)

@dataclass
class Args:
    robot: str | None = None
    """Robot to run. If not specified, lists available robots."""

    kp: float = 220.0
    """Proportional gain."""

    kd: float = 5.0
    """Derivative gain."""

    duration: float = 5.0
    """Duration of the sit-stand-sit cycle in seconds."""

    repeat: bool = False
    """Repeat the sit-stand-sit cycle continuously."""


def get_keyframe_ctrl(model: mujoco.MjModel, name: str) -> np.ndarray:
    """Get ctrl values from a named keyframe, falling back to qpos extraction."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id < 0:
        print(f"Error: keyframe '{name}' not found.")
        sys.exit(1)
    ctrl = model.key_ctrl[key_id]
    if np.any(ctrl != 0):
        return ctrl.copy()
    # Fallback: extract actuated joint positions from key_qpos.
    pose = np.zeros(model.nu)
    qpos = model.key_qpos[key_id]
    for i in range(model.nu):
        jnt_id = model.actuator_trnid[i, 0]
        pose[i] = qpos[model.jnt_qposadr[jnt_id]]
    return pose


def main() -> None:
    args = tyro.cli(Args)

    if args.robot is None:
        print("Available robots:")
        for name in ROBOT_CHOICES:
            print(f"  {name}")
        print(f"\nUsage: uv run sit_stand.py --robot <name>")
        sys.exit(0)

    if args.robot not in ROBOT_CHOICES:
        print(f"Unknown robot: {args.robot}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(ROBOTS_DIR / args.robot / "scene.xml"))
    data = mujoco.MjData(model)

    pose_sit = get_keyframe_ctrl(model, "sit")
    pose_stand = get_keyframe_ctrl(model, "stand")

    # Per-actuator PD gains scaled by ctrlrange.
    gain_scale = np.array([model.actuator_ctrlrange[i, 1] for i in range(model.nu)]) / 80.0
    kp = args.kp * gain_scale
    kd = args.kd * gain_scale

    # Build actuator -> joint index mapping.
    qpos_idx = [model.jnt_qposadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]
    dof_idx = [model.jnt_dofadr[model.actuator_trnid[i, 0]] for i in range(model.nu)]

    # Reset to sit keyframe.
    mujoco.mj_resetDataKeyframe(model, data, mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "sit"))
    mujoco.mj_forward(model, data)

    print(f"Running sit -> stand -> sit for {args.robot} ({args.duration}s, kp={args.kp}, kd={args.kd})")

    def ctrl_callback(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        t = d.time % args.duration if args.repeat else d.time
        half = args.duration / 2.0
        if t < half:
            alpha = t / half
        elif t < args.duration:
            alpha = 1.0 - (t - half) / half
        else:
            alpha = 0.0
        alpha = 0.5 * (1.0 - np.cos(np.pi * alpha))
        q_target = (1.0 - alpha) * pose_sit + alpha * pose_stand

        for i in range(m.nu):
            d.ctrl[i] = kp[i] * (q_target[i] - d.qpos[qpos_idx[i]]) - kd[i] * d.qvel[dof_idx[i]]

    mujoco.set_mjcb_control(ctrl_callback)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall_start = time.time()
        last_render = 0.0

        while viewer.is_running():
            mujoco.mj_step(model, data)

            # Sync to real time.
            sleep_time = data.time - (time.time() - wall_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Sync viewer at ~30 fps.
            now = time.time()
            if now - last_render >= 1.0 / 30.0:
                viewer.sync()
                last_render = now


if __name__ == "__main__":
    main()
