"""
Viser viewer for a generated grasp dataset.

Two modes, switched from the GUI:

  Browse    -- step through the dataset one grasp at a time. Shows the hand, the
               object at its stored pose, the target contact points (on the
               object, red) and the realised contact points (on the hand, blue).

  Neighbors -- the k-nearest-neighbour mechanism that `grasp_rrt_expand.py` uses.
               Pick a source grasp, rank a random subset of the dataset by the
               RRT distance metric, and scrub the interpolation between the
               source and the selected neighbour. "Walk graph" jumps to a random
               one of the k closest, which is the RRT expansion step.

    uv run python visualize_grasp.py --robot hsl_leap \
        --object_mesh_path my_assets/objects/cube_40mm_m.stl \
        --dataset_path ./outputs/hsl_leap_cube_40mm_rrt
"""

import argparse
import os
import threading
import time

import numpy as np
import torch
import viser
import yourdfpy
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from lygra.kinematics import build_kinematics_tree
from lygra.robot import build_robot
from lygra.utils.dataset_utils import load_grasp_dataset
from lygra.utils.geom_utils import MeshObject
from rrt_utils import interpolate_state_tau, rank_nearest_neighbors


def pose_to_viser(T):
    """4x4 matrix -> (position, wxyz quaternion)."""
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()   # xyzw
    return T[:3, 3], np.array([quat[3], quat[0], quat[1], quat[2]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Parquet file, save_to_disk directory, or Hub dataset id')
    parser.add_argument('--robot', type=str, default="hsl_leap", help='Robot name')
    parser.add_argument('--object_mesh_path', type=str, required=True, help='Path to the object mesh')
    parser.add_argument('--mode', type=str, default='Browse', choices=('Browse', 'Neighbors'),
                        help='Which panel to open on; switchable in the GUI either way')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_grasp_dataset(args.dataset_path).with_format("torch")
    print(f"Dataset loaded. Total grasps: {len(dataset)}")
    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    robot = build_robot(args.robot)
    tree = build_kinematics_tree(robot.urdf_path, robot.get_active_joints())
    urdf = yourdfpy.URDF.load(robot.urdf_path)

    # `q` is stored in get_active_joints() order; ViserUrdf and yourdfpy both want
    # URDF actuated-joint order. These are not the same list in general.
    active_joints = robot.get_active_joints()
    urdf_order = [n for n in urdf.actuated_joint_names]
    remap = [active_joints.index(n) for n in urdf_order]

    # Whole-dataset tensors for the NN search.
    dataset_q = dataset['q']
    dataset_p = dataset['object_pose']

    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"Viser server started at http://{args.host}:{args.port}")

    # Three hand instances: source/browsed (ghost), neighbour, interpolated.
    roots = {}
    urdfs = {}
    objects = {}
    for name, color in (("main", None), ("nn", (0.35, 0.55, 0.95)), ("interp", (0.95, 0.65, 0.25))):
        roots[name] = server.scene.add_frame(f"/{name}", show_axes=False)
        urdfs[name] = ViserUrdf(
            server, urdf_or_path=urdf, root_node_name=f"/{name}/robot",
            load_meshes=True, load_collision_meshes=False, mesh_color_override=color,
        )

    if not os.path.exists(args.object_mesh_path):
        raise FileNotFoundError(f"Object mesh not found: {args.object_mesh_path}")
    object_mesh = MeshObject(args.object_mesh_path).mesh
    for name in roots:
        objects[name] = server.scene.add_mesh_trimesh(f"/{name}/object", mesh=object_mesh)

    server.scene.add_frame("/base_frame", axes_length=0.05, axes_radius=0.0015)
    grid = server.scene.add_grid("grid", width=0.6, height=0.6, cell_size=0.05)

    # ------------------------------------------------------------------ GUI
    mode = server.gui.add_dropdown("Mode", ("Browse", "Neighbors"), initial_value=args.mode)

    with server.gui.add_folder("Grasp") as grasp_folder:
        index_slider = server.gui.add_slider("Index", min=0, max=len(dataset) - 1, step=1, initial_value=0)
        prev_btn = server.gui.add_button("Previous")
        next_btn = server.gui.add_button("Next")
        play_btn = server.gui.add_button("Play")
        stop_btn = server.gui.add_button("Stop")
        stop_btn.visible = False

    with server.gui.add_folder("Neighbors") as nn_folder:
        downsample_slider = server.gui.add_slider(
            "Search subset", min=min(100, len(dataset)), max=len(dataset),
            step=100, initial_value=min(2048, len(dataset)),
        )
        rank_slider = server.gui.add_slider("Neighbor rank", min=0, max=max(len(dataset) - 1, 1),
                                            step=1, initial_value=0)
        cost_display = server.gui.add_text("Distance", "0.0")
        interp_slider = server.gui.add_slider("Interpolation t", min=0.0, max=1.0, step=0.01, initial_value=0.0)
        k_slider = server.gui.add_slider("k", min=1, max=100, step=1, initial_value=10)
        walk_btn = server.gui.add_button("Walk graph (step to random k-NN)")
    nn_folder.visible = False

    with server.gui.add_folder("Show"):
        show_hand = server.gui.add_checkbox("Hand", True)
        show_object = server.gui.add_checkbox("Object", True)
        show_target = server.gui.add_checkbox("Target contacts (object)", True)
        show_contact = server.gui.add_checkbox("Realised contacts (hand)", True)
        show_grid = server.gui.add_checkbox("Grid", True)

    state = {"sorted_idx": None, "sorted_dist": None, "playing": False}
    markers = []

    def clear_markers():
        for h in markers:
            h.remove()
        markers.clear()

    def set_state(name, q, T_object):
        """Pose one hand + object instance. q is in active-joint order."""
        urdfs[name].update_cfg(np.asarray(q, dtype=np.float64)[remap])
        pos, wxyz = pose_to_viser(np.asarray(T_object, dtype=np.float64))
        objects[name].position = pos
        objects[name].wxyz = wxyz

    def draw_contacts(sample, q):
        """Target points live in the hand base frame; contact points live in the
        frame of the link named by contact_link_id, so they need FK first."""
        clear_markers()

        if show_target.value and 'target_pos' in sample:
            target_pos = sample['target_pos'].numpy()
            markers.append(server.scene.add_point_cloud(
                "/main/target_pos", points=target_pos,
                colors=np.tile([255, 40, 40], (len(target_pos), 1)).astype(np.uint8), point_size=0.004))
            if 'target_normal' in sample:
                n = sample['target_normal'].numpy()
                seg = np.stack([target_pos, target_pos + n * 0.02], axis=1)
                markers.append(server.scene.add_line_segments(
                    "/main/target_normal", points=seg, colors=(255, 40, 40), line_width=2.0))

        if show_contact.value and 'contact_pos' in sample and 'contact_link_id' in sample:
            urdf.update_cfg(np.asarray(q, dtype=np.float64)[remap])
            pos, nrm = [], []
            for p_link, n_link, link_id in zip(sample['contact_pos'].numpy(),
                                               sample['contact_normal'].numpy(),
                                               sample['contact_link_id'].numpy()):
                link_id = int(link_id)
                if not (0 <= link_id < len(tree.links)):
                    continue
                link_name = tree.links[link_id]
                if link_name not in urdf.link_map:
                    continue
                T = urdf.get_transform(link_name, urdf.base_link)
                pos.append(T[:3, :3] @ p_link + T[:3, 3])
                nrm.append(T[:3, :3] @ n_link)
            if pos:
                pos, nrm = np.array(pos), np.array(nrm)
                markers.append(server.scene.add_point_cloud(
                    "/main/contact_pos", points=pos,
                    colors=np.tile([40, 80, 255], (len(pos), 1)).astype(np.uint8), point_size=0.004))
                seg = np.stack([pos, pos + nrm * 0.02], axis=1)
                markers.append(server.scene.add_line_segments(
                    "/main/contact_normal", points=seg, colors=(40, 80, 255), line_width=2.0))

    def run_nn_search():
        idx = int(index_slider.value)
        subset = min(int(downsample_slider.value), len(dataset))
        pool = np.random.choice(len(dataset), size=subset, replace=False)
        # Always keep the source in the pool so rank 0 is itself at distance 0.
        if idx not in pool:
            pool[0] = idx

        sorted_local, sorted_dist = rank_nearest_neighbors(
            dataset_q[idx], dataset_p[idx], dataset_q[pool], dataset_p[pool]
        )
        state["sorted_idx"] = pool[sorted_local.numpy()]
        state["sorted_dist"] = sorted_dist.numpy()

        rank_slider.max = subset - 1
        if rank_slider.value > subset - 1:
            rank_slider.value = subset - 1

    def update():
        is_nn = mode.value == "Neighbors"
        grasp_folder.visible = True
        nn_folder.visible = is_nn
        roots["nn"].visible = is_nn and show_hand.value
        roots["interp"].visible = is_nn and show_hand.value
        roots["main"].visible = show_hand.value
        for name in objects:
            objects[name].visible = show_object.value and (name == "main" or is_nn)

        idx = int(index_slider.value)
        sample = dataset[idx]
        q_src, p_src = sample['q'], sample['object_pose']
        set_state("main", q_src, p_src)
        draw_contacts(sample, q_src)

        if not is_nn:
            return

        if state["sorted_idx"] is None:
            run_nn_search()

        rank = int(rank_slider.value)
        nn_idx = int(state["sorted_idx"][rank])
        cost_display.value = f"{state['sorted_dist'][rank]:.4f}"

        nn_sample = dataset[nn_idx]
        q_nn, p_nn = nn_sample['q'], nn_sample['object_pose']
        set_state("nn", q_nn, p_nn)

        q_i, p_i = interpolate_state_tau(q_src, p_src, q_nn, p_nn, tau=float(interp_slider.value))
        set_state("interp", q_i, p_i)

    # ------------------------------------------------------------- callbacks
    @index_slider.on_update
    def _(_):
        if mode.value == "Neighbors":
            run_nn_search()
        update()

    @mode.on_update
    def _(_):
        if mode.value == "Neighbors":
            run_nn_search()
        update()

    @downsample_slider.on_update
    def _(_):
        run_nn_search()
        update()

    for widget in (rank_slider, interp_slider, show_hand, show_object, show_target, show_contact):
        widget.on_update(lambda _: update())

    @show_grid.on_update
    def _(_):
        grid.visible = show_grid.value

    @prev_btn.on_click
    def _(_):
        index_slider.value = max(0, index_slider.value - 1)

    @next_btn.on_click
    def _(_):
        index_slider.value = min(len(dataset) - 1, index_slider.value + 1)

    @walk_btn.on_click
    def _(_):
        if state["sorted_idx"] is None:
            return
        # rank 0 is the source itself; step to one of the k closest others.
        k = min(int(k_slider.value), len(state["sorted_idx"]) - 1)
        if k < 1:
            return
        index_slider.value = int(state["sorted_idx"][np.random.randint(1, k + 1)])
        rank_slider.value = 0

    @play_btn.on_click
    def _(_):
        state["playing"] = True
        play_btn.visible, stop_btn.visible = False, True

    @stop_btn.on_click
    def _(_):
        state["playing"] = False
        play_btn.visible, stop_btn.visible = True, False

    def playback_loop():
        while True:
            if state["playing"]:
                if index_slider.value < len(dataset) - 1:
                    index_slider.value += 1
                else:
                    state["playing"] = False
                    play_btn.visible, stop_btn.visible = True, False
                time.sleep(0.3)
            else:
                time.sleep(0.1)

    threading.Thread(target=playback_loop, daemon=True).start()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.3, 0.3, 0.3)
        client.camera.look_at = (0.0, 0.0, 0.0)

    update()

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
