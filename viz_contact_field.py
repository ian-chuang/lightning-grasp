"""
Viser inspector for a robot's contact-field configuration.

Renders, on top of the posed hand:
  * green arrows  -- surface patches KEPT by the `disabled_normal` rules
                     (these are the movable contact patches the sampler uses)
  * red arrows    -- surface patches DROPPED by those rules
  * blue arrows   -- static patches kept by the `allowed_normal` rules (palm)
  * green box     -- the canonical space from get_canonical_space()

Use it to check that the kept patches cover the parts of the fingertips that
actually touch objects, and that the canonical box sits where the fingers close.

This deliberately does NOT call robot.get_contact_field(): that runs 80k FK
samples on the GPU to build the swept field. Here we only need the per-link
patches in link frame, which is the cheap first half of the same pipeline, so
this runs on CPU in a few seconds.

    uv run python viz_contact_field.py --robot hsl_leap
"""

import argparse
import time

import numpy as np
import trimesh
import viser
import yourdfpy
from viser.extras import ViserUrdf

from lygra.mesh import RobotMesh
from lygra.mesh_analyzer import generate_mesh_group
from lygra.contact_field import has_aligned
from lygra.robot import build_robot


def classify_link_patches(mesh, rule, is_static, pos_tol, rot_tol):
    """Run the same patch grouping + normal filtering as build_contact_field().

    Returns (kept, dropped), each an [N, 6] array of (origin, normal) rows in
    the link's own frame.
    """
    group = generate_mesh_group(
        mesh=mesh,
        patch_pos_dist_tol=pos_tol,
        patch_rot_dist_tol=rot_tol,
    )

    kept, dropped = [], []
    for keyvector, centroid in zip(group["group_vectors"], group["group_centroid"]):
        if is_static:
            # static: keep the patch when its centroid normal is inside one of
            # the allowed cones.
            passed = any(
                has_aligned(centroid[:, 3:], allowed, threshold=threshold)
                for allowed, threshold in rule["allowed_normal"]
            )
        else:
            # movable: drop the patch when any key vector falls inside a
            # disabled cone.
            passed = not any(
                has_aligned(keyvector[:, 3:], disabled, threshold=threshold)
                for disabled, threshold in rule["disabled_normal"]
            )

        (kept if passed else dropped).append(keyvector)

    def stack(x):
        return np.concatenate(x) if len(x) else np.zeros((0, 6))

    return stack(kept), stack(dropped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robot', type=str, default='hsl_leap', help='Robot name')
    parser.add_argument('--urdf_path', type=str, default=None, help='Override URDF path')
    parser.add_argument('--arrow_length', type=float, default=0.008)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    robot = build_robot(args.robot, urdf_path=args.urdf_path)
    config = robot.get_contact_field_config()
    pos_tol = config.get("patch_pos_dist_tol", 0.01)
    rot_tol = config.get("patch_rot_dist_tol", 3.1415 / 5)

    urdf = yourdfpy.URDF.load(robot.urdf_path)
    robot_mesh = RobotMesh(robot.urdf_path, mesh_scale=robot.get_mesh_scale())

    active_joints = robot.get_active_joints()
    print(f"robot        : {args.robot}")
    print(f"urdf         : {robot.urdf_path}")
    print(f"root link    : {urdf.base_link}")
    print(f"active joints: {active_joints}")

    # ------------------------------------------------------------------
    # Patch classification (link frame, done once)
    # ------------------------------------------------------------------
    link_patches = {}
    for link_name, rule in list(config["movable_link"].items()) + list(config["static_link"].items()):
        is_static = link_name in config["static_link"]

        if link_name not in urdf.link_map:
            print(f"  [!] '{link_name}' is not a link in this URDF -- rule has no effect")
            continue

        mesh = robot_mesh.get_link_collision_mesh(link_name)
        if mesh.vertices.shape[0] == 0:
            print(f"  [!] '{link_name}' has no collision geometry -- rule has no effect")
            continue

        kept, dropped = classify_link_patches(mesh, rule, is_static, pos_tol, rot_tol)
        link_patches[link_name] = (kept, dropped, is_static)
        total = len(kept) + len(dropped)
        kind = "static " if is_static else "movable"
        print(f"  {kind} {link_name:12s} kept {len(kept):5d} / {total:5d} key vectors "
              f"({100.0 * len(kept) / max(total, 1):.0f}%)")

    if not link_patches:
        print("No configured link produced any patch. Check the link names in "
              "get_contact_field_config() against the URDF.")

    # ------------------------------------------------------------------
    # Viser scene
    # ------------------------------------------------------------------
    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"\nViser running at http://{args.host}:{args.port}")

    viser_urdf = ViserUrdf(server, urdf_or_path=urdf, load_meshes=True, load_collision_meshes=False)
    server.scene.add_frame("/root_frame", axes_length=0.05, axes_radius=0.0015)
    server.scene.add_grid("grid", width=0.6, height=0.6, cell_size=0.05)

    with server.gui.add_folder("Show"):
        show_kept = server.gui.add_checkbox("Kept patches (green)", True)
        show_dropped = server.gui.add_checkbox("Dropped patches (red)", False)
        show_static = server.gui.add_checkbox("Static patches (blue)", True)
        show_mesh = server.gui.add_checkbox("Hand mesh", True)
        show_box = server.gui.add_checkbox("Canonical space", True)

    box_min, box_max = robot.get_canonical_space()
    box_sliders = []
    with server.gui.add_folder("Canonical space"):
        for i, axis in enumerate("xyz"):
            box_sliders.append((
                server.gui.add_slider(f"{axis} min", min=-0.25, max=0.25, step=0.001,
                                      initial_value=float(box_min[i])),
                server.gui.add_slider(f"{axis} max", min=-0.25, max=0.25, step=0.001,
                                      initial_value=float(box_max[i])),
            ))
        print_box = server.gui.add_button("Print box")

    joint_sliders = []
    with server.gui.add_folder("Joints"):
        reset_btn = server.gui.add_button("Reset to zero")
        for name in active_joints:
            limit = urdf.joint_map[name].limit
            lower, upper = float(limit.lower), float(limit.upper)
            joint_sliders.append(server.gui.add_slider(
                name, min=lower, max=upper, step=0.01,
                initial_value=float(np.clip(0.0, lower, upper)),   # some limits exclude zero
            ))

    # ViserUrdf expects values in its own actuated-joint order, which is URDF
    # order, not robot.get_active_joints() order.
    urdf_order = viser_urdf.get_actuated_joint_names()
    remap = [active_joints.index(n) for n in urdf_order]

    arrow_handles = []

    def draw_patches():
        for h in arrow_handles:
            h.remove()
        arrow_handles.clear()

        for link_name, (kept, dropped, is_static) in link_patches.items():
            T = urdf.get_transform(link_name, urdf.base_link)

            groups = []
            if is_static:
                groups.append(("static", kept, (60, 120, 255), show_static.value))
            else:
                groups.append(("kept", kept, (40, 200, 60), show_kept.value))
            groups.append(("dropped", dropped, (220, 50, 50), show_dropped.value))

            for tag, vecs, color, visible in groups:
                if len(vecs) == 0:
                    continue
                origin = vecs[:, :3] @ T[:3, :3].T + T[:3, 3]
                normal = vecs[:, 3:] @ T[:3, :3].T
                segments = np.stack([origin, origin + normal * args.arrow_length], axis=1)
                handle = server.scene.add_line_segments(
                    f"/patches/{link_name}/{tag}",
                    points=segments.astype(np.float32),
                    colors=np.tile(np.array(color, dtype=np.uint8), (len(segments), 2, 1)),
                    line_width=2.0,
                )
                handle.visible = visible
                arrow_handles.append(handle)

    box_handle = None

    def draw_box():
        nonlocal box_handle
        if box_handle is not None:
            box_handle.remove()
        lo = np.array([s[0].value for s in box_sliders])
        hi = np.array([s[1].value for s in box_sliders])
        extents = np.maximum(hi - lo, 1e-4)
        mesh = trimesh.creation.box(extents=extents)
        mesh.visual.face_colors = [0, 255, 0, 70]
        box_handle = server.scene.add_mesh_trimesh(
            "/canonical_space", mesh=mesh, position=(lo + hi) / 2.0
        )
        box_handle.visible = show_box.value

    def update_pose(_=None):
        q = np.array([s.value for s in joint_sliders])
        viser_urdf.update_cfg(q[remap])
        urdf.update_cfg(q[remap])
        draw_patches()

    for slider in joint_sliders:
        slider.on_update(update_pose)

    for lo_s, hi_s in box_sliders:
        lo_s.on_update(lambda _: draw_box())
        hi_s.on_update(lambda _: draw_box())

    @print_box.on_click
    def _(_):
        lo = np.array([s[0].value for s in box_sliders])
        hi = np.array([s[1].value for s in box_sliders])
        print(f"box_min = np.array({np.round(lo, 4).tolist()}, dtype=np.float32)")
        print(f"box_max = np.array({np.round(hi, 4).tolist()}, dtype=np.float32)")

    @reset_btn.on_click
    def _(_):
        for slider in joint_sliders:
            slider.value = float(np.clip(0.0, slider.min, slider.max))

    for checkbox in (show_kept, show_dropped, show_static):
        checkbox.on_update(lambda _: draw_patches())

    @show_mesh.on_update
    def _(_):
        for frame in viser_urdf._joint_frames:
            frame.visible = show_mesh.value

    @show_box.on_update
    def _(_):
        box_handle.visible = show_box.value

    update_pose()
    draw_box()

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
