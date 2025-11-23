import time
import argparse
from typing import Literal, Optional, Tuple
import numpy as np
import torch
import trimesh
import viser
from viser.extras import ViserUrdf
from datasets import load_dataset, load_from_disk
import os
from scipy.spatial.transform import Rotation
import yourdfpy

# Lygra imports
try:
    from lygra.robot import build_robot
    from lygra.utils.geom_utils import MeshObject
    from lygra.kinematics import build_kinematics_tree
except ImportError:
    print("Lygra package not found. Please ensure you are in the correct environment.")
    exit(1)

from rrt_utils import sample_random_q, sample_random_object_pose, interpolate_state, rank_nearest_neighbors, interpolate_state_tau

def load_grasp_dataset(data_path):
    """
    Load the dataset from parquet or directory.
    """
    if data_path.endswith('.parquet'):
        dataset = load_dataset("parquet", data_files=data_path, split="train")
    else:
        try:
            dataset = load_dataset(data_path, split="train")
        except:
             dataset = load_from_disk(data_path)
    
    dataset = dataset.with_format("torch")
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Visualize Nearest Neighbor Search and RRT")
    parser.add_argument('--dataset_path', type=str, default="iantc104/leap_hand_grasp_cube_rrt", help='Path to the generated dataset file (parquet) or HF Hub ID')
    parser.add_argument('--robot', type=str, default="leap", help='Robot Name')
    parser.add_argument('--object_mesh_path', type=str, default="./assets/40mm_cube.stl", help='Path to the object mesh')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for Viser server')
    parser.add_argument('--port', type=int, default=8080, help='Port for Viser server')
    parser.add_argument('--num_queries', type=int, default=50, help='Number of random queries to pre-generate')
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_grasp_dataset(args.dataset_path)
    print(f"Dataset loaded. Total grasps: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    # Pre-load dataset tensors
    dataset_q = dataset['q']
    if not isinstance(dataset_q, torch.Tensor):
        dataset_q = torch.tensor(np.array(dataset_q))
        
    dataset_p = dataset['object_pose']
    if not isinstance(dataset_p, torch.Tensor):
        dataset_p = torch.tensor(np.array(dataset_p))
    
    # Start Viser server
    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"Viser server started at http://{args.host}:{args.port}")

    # Robot orientation
    ROBOT_ROTATION = np.array([-0.707, 0, 0.707, 0])
    
    # Create root frames
    def create_robot_root(name, visible=True):
        return server.scene.add_frame(
            name,
            wxyz=ROBOT_ROTATION,
            position=(0, 0, 0),
            show_axes=False,
            visible=visible
        )

    root_query = create_robot_root("/root_query", visible=True)
    root_nn = create_robot_root("/root_nn", visible=True)
    root_interp = create_robot_root("/root_interp", visible=True)

    # State variables
    current_robot_name = None
    current_mesh_path = None
    
    # URDF loaders
    urdf_query: Optional[ViserUrdf] = None
    urdf_nn: Optional[ViserUrdf] = None
    urdf_interp: Optional[ViserUrdf] = None
    
    # Object handles
    obj_query = None
    obj_nn = None
    obj_interp = None
    
    # Data
    lygra_robot = None
    
    # Current State
    current_query_idx = 0
    current_nn_rank = 0
    current_downsample_size = 2048
    current_sorted_indices = None
    current_sorted_dists = None
    
    # GUI elements
    with server.gui.add_folder("Source Selection"):
        query_slider = server.gui.add_slider(
            "Dataset Index (Source)",
            min=0,
            max=len(dataset) - 1,
            step=1,
            initial_value=0,
        )

    with server.gui.add_folder("NN Search"):
        downsample_slider = server.gui.add_slider(
            "Downsample Size",
            min=100,
            max=len(dataset),
            step=100,
            initial_value=min(2048, len(dataset)),
        )
        
        nn_rank_slider = server.gui.add_slider(
            "NN Rank (Cost)",
            min=0,
            max=min(2048, len(dataset)) - 1,
            step=1,
            initial_value=0,
        )
        
        cost_display = server.gui.add_text("NN Cost", "0.0")

        k_slider = server.gui.add_slider(
            "k (Random Sample)",
            min=1,
            max=100,
            step=1,
            initial_value=10,
        )
        sample_btn = server.gui.add_button("Step to Random Neighbor (Walk Graph)")

    with server.gui.add_folder("Interpolation"):
        interp_slider = server.gui.add_slider(
            "t",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=0.0,
        )

    with server.gui.add_folder("Visualization Options"):
        show_query = server.gui.add_checkbox("Show Source (Ghost)", True)
        show_nn = server.gui.add_checkbox("Show Neighbor (Solid)", True)
        show_interp = server.gui.add_checkbox("Show Interpolated", True)
        show_contacts = server.gui.add_checkbox("Show Contacts", True)

    def load_robot_and_object(robot_name, mesh_path):
        nonlocal current_robot_name, current_mesh_path, lygra_robot
        nonlocal urdf_query, urdf_nn, urdf_interp, obj_query, obj_nn, obj_interp
        
        if robot_name != current_robot_name:
            print(f"Loading robot: {robot_name}")
            lygra_robot = build_robot(robot_name)
            urdf_path = lygra_robot.urdf_path
            urdf_model = yourdfpy.URDF.load(urdf_path)
            
            urdf_query = ViserUrdf(server, urdf_or_path=urdf_model, root_node_name="/root_query/robot")
            urdf_nn = ViserUrdf(server, urdf_or_path=urdf_model, root_node_name="/root_nn/robot")
            urdf_interp = ViserUrdf(server, urdf_or_path=urdf_model, root_node_name="/root_interp/robot")
            
            current_robot_name = robot_name

        if mesh_path != current_mesh_path:
            print(f"Loading object: {mesh_path}")
            if os.path.exists(mesh_path):
                mesh_obj = MeshObject(mesh_path)
                
                def create_obj(name, root):
                    return server.scene.add_mesh_trimesh(
                        name=f"{root}/{name}",
                        mesh=mesh_obj.mesh,
                        position=(0,0,0),
                        wxyz=(1,0,0,0)
                    )
                
                obj_query = create_obj("object", "/root_query")
                obj_nn = create_obj("object", "/root_nn")
                obj_interp = create_obj("object", "/root_interp")
                
                current_mesh_path = mesh_path

    def update_nn_search():
        nonlocal current_sorted_indices, current_sorted_dists
        
        q_source = dataset_q[current_query_idx]
        p_source = dataset_p[current_query_idx]
        
        # Downsample dataset
        N = len(dataset)
        sample_size = min(current_downsample_size, N)
        indices = np.random.choice(N, size=sample_size, replace=False)
        
        ds_q = dataset_q[indices]
        ds_p = dataset_p[indices]
        
        # Rank NNs
        sorted_idx_local, sorted_dists = rank_nearest_neighbors(q_source, p_source, ds_q, ds_p)
        
        # Map back to original dataset indices
        current_sorted_indices = indices[sorted_idx_local.cpu().numpy()]
        current_sorted_dists = sorted_dists.cpu().numpy()
        
        # Update slider max
        nn_rank_slider.max = sample_size - 1
        if nn_rank_slider.value >= sample_size:
            nn_rank_slider.value = sample_size - 1

    def update_visualization():
        if current_sorted_indices is None: return
        
        # Visibility
        root_query.visible = show_query.value
        root_nn.visible = show_nn.value
        root_interp.visible = show_interp.value
        
        # 1. Source State
        q_source = dataset_q[current_query_idx]
        p_source = dataset_p[current_query_idx]
        update_pose(urdf_query, obj_query, q_source, p_source)
        
        # 2. NN State
        rank = int(nn_rank_slider.value)
        idx = current_sorted_indices[rank]
        dist = current_sorted_dists[rank]
        cost_display.value = f"{dist:.4f}"
        
        sample_nn = dataset[int(idx)]
        q_nn = sample_nn['q']
        p_nn = sample_nn['object_pose']
        
        update_pose(urdf_nn, obj_nn, q_nn, p_nn)
        
        # 3. Interpolated State
        t = interp_slider.value
        # Interpolate from Source (t=0) to Neighbor (t=1)
        q_interp, p_interp = interpolate_state_tau(q_source, p_source, q_nn, p_nn, tau=t)
        update_pose(urdf_interp, obj_interp, q_interp, p_interp)
        
        # 4. Contacts (Interpolated)
        if show_contacts.value:
            visualize_contacts(sample_nn, p_nn, p_interp)

    def update_pose(urdf_viz, obj_handle, q, p):
        if urdf_viz and q is not None:
            urdf_viz.update_cfg(q.flatten().cpu().numpy())
        
        if obj_handle and p is not None:
            p_np = p.detach().cpu().numpy()
            R = p_np[:3, :3]
            t = p_np[:3, 3]
            rot = Rotation.from_matrix(R)
            quat = rot.as_quat()
            wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
            obj_handle.position = t
            obj_handle.wxyz = wxyz

    def visualize_contacts(sample, p_nn, p_interp):
        # Clear previous? Viser handles usually persist unless removed.
        # We reuse names so they get overwritten.
        
        if 'target_pos' in sample:
            target_pos = sample['target_pos']
            if not isinstance(target_pos, torch.Tensor): target_pos = torch.tensor(target_pos)
            target_pos = target_pos.detach().cpu()
            
            p_nn = p_nn.detach().cpu()
            p_interp = p_interp.detach().cpu()
            
            n_contact = target_pos.shape[0]
            target_pos_homog = torch.cat([target_pos, torch.ones((n_contact, 1))], dim=1)
            
            p_nn_inv = torch.linalg.inv(p_nn)
            target_pos_obj = (p_nn_inv @ target_pos_homog.T).T
            
            target_pos_interp_homog = (p_interp @ target_pos_obj.T).T
            target_pos_interp = target_pos_interp_homog[:, :3].numpy()

            server.scene.add_point_cloud(
                name="/root_interp/target_pos",
                points=target_pos_interp,
                colors=np.array([[1.0, 0.0, 0.0]] * len(target_pos_interp)),
                point_size=0.005
            )

            if 'target_normal' in sample:
                target_normal = sample['target_normal']
                if not isinstance(target_normal, torch.Tensor): target_normal = torch.tensor(target_normal)
                target_normal = target_normal.detach().cpu()
                
                R_nn = p_nn[:3, :3]
                R_interp = p_interp[:3, :3]
                R_rel = R_interp @ R_nn.T
                
                target_normal_interp = (R_rel @ target_normal.T).T.numpy()
                
                scale = 0.03
                lines = []
                for pos, normal in zip(target_pos_interp, target_normal_interp):
                    lines.append([pos, pos + normal * scale])
                lines = np.array(lines)

                server.scene.add_line_segments(
                    name="/root_interp/target_normal",
                    points=lines,
                    colors=(0.0, 1.0, 0.0),
                    line_width=2.0
                )

    # Callbacks
    @query_slider.on_update
    def _(_):
        nonlocal current_query_idx
        current_query_idx = int(query_slider.value)
        update_nn_search()
        update_visualization()
        
    @sample_btn.on_click
    def _(_):
        k = int(k_slider.value)
        max_rank = int(nn_rank_slider.max)
        effective_k = min(k, max_rank + 1)
        
        random_rank = np.random.randint(0, effective_k)
        
        if current_sorted_indices is not None:
            # Get the dataset index of the selected neighbor
            new_source_idx = current_sorted_indices[random_rank]
            
            # Update the source slider to this new index
            # This triggers the query_slider callback, which updates the search and visualization
            query_slider.value = int(new_source_idx)
            
            # Reset rank to 0 so we start focused on the new source
            nn_rank_slider.value = 0

    @downsample_slider.on_update
    def _(_):
        nonlocal current_downsample_size
        current_downsample_size = int(downsample_slider.value)
        update_nn_search()
        update_visualization()
        
    @nn_rank_slider.on_update
    def _(_):
        update_visualization()
        
    @interp_slider.on_update
    def _(_):
        update_visualization()
        
    @show_query.on_update
    def _(_): update_visualization()
    @show_nn.on_update
    def _(_): update_visualization()
    @show_interp.on_update
    def _(_): update_visualization()
    @show_contacts.on_update
    def _(_): update_visualization()

    # Initial Setup
    if len(dataset) > 0:
        load_robot_and_object(args.robot, args.object_mesh_path)
        update_nn_search()
        update_visualization()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.3, 0.3, 0.3)
        client.camera.look_at = (0.0, 0.0, 0.0)

    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
