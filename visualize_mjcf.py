#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro
import viser
from viser.extras import ViserUrdf

ROBOTS_DIR = Path(__file__).parent / "robots"

ROBOT_CHOICES = sorted(
    d.name for d in ROBOTS_DIR.iterdir() if d.is_dir() and list(d.glob("scene.xml"))
)

# URDF paths for viser visualisation (auto-discovered from tbai_descriptions submodule).
URDF_DIR = (Path(__file__).parent / "thirdparty" / "tbai_descriptions" / "robots").resolve()
ROBOT_URDFS: dict[str, Path] = {
    p.stem: p
    for d in URDF_DIR.iterdir()
    if d.is_dir()
    for p in [d / f"{d.name}.urdf"]
    if p.exists()
}

# RGB colors for frame axes.
AXIS_COLORS = [
    (1.0, 0.0, 0.0, 1.0),  # X = red
    (0.0, 1.0, 0.0, 1.0),  # Y = green
    (0.0, 0.0, 1.0, 1.0),  # Z = blue
]


def _add_body_frame_geoms(
    scn: mujoco.MjvScene,
    pos: np.ndarray,
    rot: np.ndarray,
    length: float,
    width: float,
) -> None:
    """Add three axis arrows to user_scn for a single body frame."""
    for ax in range(3):
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        # Arrow: size = [shaft_radius, head_radius, length].
        # Arrow points along +Z in its local frame.
        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=np.array([width, width * 2, length]),
            pos=pos.copy(),
            mat=np.eye(3).flatten(),
            rgba=np.array(AXIS_COLORS[ax], dtype=np.float32),
        )
        # Build rotation: map local +Z to the desired axis direction.
        z = rot[:, ax].copy()
        up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        g.mat[:] = np.column_stack([x, y, z])
        scn.ngeom += 1


@dataclass
class Args:
    """MuJoCo robot visualizer for tbai_mujoco_descriptions robots."""

    robot: str | None = None
    """Robot to visualize. If not specified, lists available robots."""
    no_browser: bool = False
    """Do not open the browser automatically."""


def main() -> None:
    args = tyro.cli(Args)

    if args.robot is None:
        print("Available robots:")
        for name in ROBOT_CHOICES:
            print(f"  {name}")
        print(f"\nUsage: uv run visualize_mjcf.py --robot <name>")
        sys.exit(0)

    if args.robot not in ROBOT_CHOICES:
        print(f"Unknown robot: {args.robot}")
        print("Available robots:")
        for name in ROBOT_CHOICES:
            print(f"  {name}")
        sys.exit(1)

    # --- Load MuJoCo model ---
    scene_path = ROBOTS_DIR / args.robot / "scene.xml"
    mj_model = mujoco.MjModel.from_xml_path(str(scene_path))
    mj_data = mujoco.MjData(mj_model)

    # Collect controllable MuJoCo joints (skip free joints).
    mj_joints: list[dict] = []
    for i in range(mj_model.njnt):
        if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
        qpos_adr = mj_model.jnt_qposadr[i]
        mj_joints.append(dict(name=name, qpos_adr=qpos_adr))

    # Collect all MuJoCo bodies (skip world body 0).
    mj_bodies: list[dict] = []
    for i in range(1, mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        mj_bodies.append(dict(id=i, name=name))

    # Collect all MuJoCo sites.
    mj_sites: list[dict] = []
    for i in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, i) or f"site_{i}"
        mj_sites.append(dict(id=i, name=name))

    mujoco.mj_forward(mj_model, mj_data)

    # --- Set up viser ---
    server = viser.ViserServer()
    if not args.no_browser:
        webbrowser.open(f"http://localhost:{server.get_port()}")

    urdf_path = ROBOT_URDFS.get(args.robot)
    if urdf_path is None or not urdf_path.exists():
        print(f"Warning: no URDF found for {args.robot}, viser will show controls only.")

    # State containers.
    current_urdf: list[ViserUrdf | None] = [None]
    current_urdf_grey: list[ViserUrdf | None] = [None]
    slider_handles: list[viser.GuiInputHandle[float]] = []
    initial_config: list[float] = []
    joint_folder_handle: list = [None]
    opacity_slider_handle: list = [None]
    frames_folder_handle: list = [None]

    # Frame controls (shared for viser + MuJoCo).
    mj_body_cbs: list[viser.GuiInputHandle[bool]] = []
    mj_site_cbs: list[viser.GuiInputHandle[bool]] = []
    viser_joint_cbs: list[viser.GuiInputHandle[bool]] = []
    show_all_cb_handle: list = [None]
    frame_size_slider_handle: list = [None]

    def load_robot() -> None:
        # Clean up previous state.
        if current_urdf[0] is not None:
            current_urdf[0].remove()
        if current_urdf_grey[0] is not None:
            current_urdf_grey[0].remove()
        slider_handles.clear()
        initial_config.clear()
        if joint_folder_handle[0] is not None:
            joint_folder_handle[0].remove()
        if opacity_slider_handle[0] is not None:
            opacity_slider_handle[0].remove()
        if frames_folder_handle[0] is not None:
            frames_folder_handle[0].remove()
        mj_body_cbs.clear()
        mj_site_cbs.clear()
        viser_joint_cbs.clear()

        # Load URDF in viser (original + grey for opacity).
        if urdf_path and urdf_path.exists():
            viser_urdf = ViserUrdf(
                server, urdf_or_path=urdf_path, root_node_name="/robot",
            )
            viser_urdf_grey = ViserUrdf(
                server, urdf_or_path=urdf_path, root_node_name="/robot_grey",
                mesh_color_override=(0.6, 0.6, 0.7, 1.0),
            )
            for mh in viser_urdf_grey._meshes:
                mh.visible = False
            current_urdf[0] = viser_urdf
            current_urdf_grey[0] = viser_urdf_grey
        else:
            current_urdf[0] = None
            current_urdf_grey[0] = None

        # --- Joint sliders ---
        folder = server.gui.add_folder("Joint controls")
        joint_folder_handle[0] = folder
        with folder:
            for idx, mj_j in enumerate(mj_joints):
                jnt_id = mj_j["qpos_adr"]
                mj_idx = None
                for i in range(mj_model.njnt):
                    if mj_model.jnt_qposadr[i] == jnt_id:
                        mj_idx = i
                        break
                has_limits = bool(mj_model.jnt_limited[mj_idx]) if mj_idx is not None else False
                lower = float(mj_model.jnt_range[mj_idx, 0]) if has_limits else -np.pi
                upper = float(mj_model.jnt_range[mj_idx, 1]) if has_limits else np.pi
                initial_pos = 0.0 if lower < -0.1 and upper > 0.1 else (lower + upper) / 2.0

                slider = server.gui.add_slider(
                    label=mj_j["name"],
                    min=lower,
                    max=upper,
                    step=1e-3,
                    initial_value=initial_pos,
                )

                def _on_update(_, _urdf=current_urdf, _urdf_grey=current_urdf_grey):
                    cfg = np.array([s.value for s in slider_handles])
                    if _urdf[0] is not None:
                        _urdf[0].update_cfg(cfg)
                    if _urdf_grey[0] is not None:
                        _urdf_grey[0].update_cfg(cfg)

                slider.on_update(_on_update)
                slider_handles.append(slider)
                initial_config.append(initial_pos)

            reset_btn = server.gui.add_button("Reset joints")

            @reset_btn.on_click
            def _(_):
                for s, q in zip(slider_handles, initial_config):
                    s.value = q

        # Set initial config.
        cfg = np.array(initial_config)
        if current_urdf[0] is not None:
            current_urdf[0].update_cfg(cfg)
        if current_urdf_grey[0] is not None:
            current_urdf_grey[0].update_cfg(cfg)

        # --- Opacity ---
        opacity_slider = server.gui.add_slider(
            label="Model opacity", min=0.0, max=1.0, step=0.01, initial_value=1.0,
        )
        opacity_slider_handle[0] = opacity_slider

        original_meshes = current_urdf[0]._meshes if current_urdf[0] else []
        grey_meshes = current_urdf_grey[0]._meshes if current_urdf_grey[0] else []

        def _apply_opacity(val: float) -> None:
            if val >= 1.0:
                for mh in original_meshes:
                    mh.visible = True
                for mh in grey_meshes:
                    mh.visible = False
            else:
                for mh in original_meshes:
                    mh.visible = False
                for mh in grey_meshes:
                    mh.visible = True
                    mh.opacity = val if val > 0.0 else 0.0

        @opacity_slider.on_update
        def _(_):
            _apply_opacity(opacity_slider.value)

        # --- Frame visualization ---
        frames_folder = server.gui.add_folder("Frame visualization")
        frames_folder_handle[0] = frames_folder

        # Viser URDF joint frames setup.
        viser_joint_names: list[str] = []
        viser_joint_frame_handles: list = []
        if current_urdf[0] is not None:
            viser_joint_names = list(current_urdf[0]._urdf.joint_map.keys())
            viser_joint_frame_handles = current_urdf[0]._joint_frames
            for fh in viser_joint_frame_handles:
                fh.axes_length = 0.05
                fh.axes_radius = 0.05 * 0.05
                fh.origin_radius = 0.05 * 0.1
                fh.show_axes = False

        with frames_folder:
            show_all_cb = server.gui.add_checkbox("Show all frames", initial_value=False)
            show_all_cb_handle[0] = show_all_cb

            frame_size_slider = server.gui.add_slider(
                label="Frame size", min=0.01, max=0.3, step=0.005, initial_value=0.05,
            )
            frame_size_slider_handle[0] = frame_size_slider

            @frame_size_slider.on_update
            def _(_):
                size = frame_size_slider.value
                for fh in viser_joint_frame_handles:
                    fh.axes_length = size
                    fh.axes_radius = size * 0.05
                    fh.origin_radius = size * 0.1

            # --- Viser URDF joint frame checkboxes ---
            if viser_joint_names:
                server.gui.add_markdown("**Viser (URDF) joint frames**")
            for i, jname in enumerate(viser_joint_names):
                cb = server.gui.add_checkbox(jname, initial_value=False)
                viser_joint_cbs.append(cb)

                def make_viser_cb_handler(idx: int, checkbox: viser.GuiInputHandle[bool]):
                    def handler(_):
                        if idx < len(viser_joint_frame_handles):
                            viser_joint_frame_handles[idx].show_axes = checkbox.value
                    return handler

                cb.on_update(make_viser_cb_handler(i, cb))

            # --- MuJoCo body frame checkboxes ---
            server.gui.add_markdown("**MuJoCo body frames**")
            for body in mj_bodies:
                cb = server.gui.add_checkbox(body["name"], initial_value=False)
                mj_body_cbs.append(cb)

            # --- MuJoCo site frame checkboxes ---
            if mj_sites:
                server.gui.add_markdown("**MuJoCo site frames**")
                for site in mj_sites:
                    cb = server.gui.add_checkbox(site["name"], initial_value=False)
                    mj_site_cbs.append(cb)

            # Show all toggles viser, MuJoCo body, and MuJoCo site frames.
            @show_all_cb.on_update
            def _(_):
                enabled = show_all_cb.value
                for i, cb in enumerate(viser_joint_cbs):
                    cb.value = enabled
                    if i < len(viser_joint_frame_handles):
                        viser_joint_frame_handles[i].show_axes = enabled
                for cb in mj_body_cbs:
                    cb.value = enabled
                for cb in mj_site_cbs:
                    cb.value = enabled

        # Ground grid.
        if current_urdf[0] is not None:
            trimesh_scene = current_urdf[0]._urdf.scene or current_urdf[0]._urdf.collision_scene
            grid_z = trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0
        else:
            grid_z = 0.0
        server.scene.add_grid("/grid", width=2, height=2, position=(0.0, 0.0, grid_z))

    # Reload button.
    reload_btn = server.gui.add_button("Reload description")

    @reload_btn.on_click
    def _(_):
        load_robot()

    load_robot()

    # --- Launch MuJoCo passive viewer ---
    with mujoco.viewer.launch_passive(mj_model, mj_data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE

        while viewer.is_running():
            # Sync slider values → MuJoCo qpos.
            for s, mj_j in zip(slider_handles, mj_joints):
                mj_data.qpos[mj_j["qpos_adr"]] = s.value

            mujoco.mj_forward(mj_model, mj_data)

            # Draw per-body and per-site frames in user_scn.
            frame_size = frame_size_slider_handle[0].value if frame_size_slider_handle[0] else 0.05
            viewer.user_scn.ngeom = 0
            for cb, body in zip(mj_body_cbs, mj_bodies):
                if not cb.value:
                    continue
                bid = body["id"]
                pos = mj_data.xpos[bid]
                rot = mj_data.xmat[bid].reshape(3, 3)
                _add_body_frame_geoms(
                    viewer.user_scn, pos, rot,
                    length=frame_size, width=frame_size * 0.02,
                )
            for cb, site in zip(mj_site_cbs, mj_sites):
                if not cb.value:
                    continue
                sid = site["id"]
                pos = mj_data.site_xpos[sid]
                rot = mj_data.site_xmat[sid].reshape(3, 3)
                _add_body_frame_geoms(
                    viewer.user_scn, pos, rot,
                    length=frame_size, width=frame_size * 0.02,
                )

            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
