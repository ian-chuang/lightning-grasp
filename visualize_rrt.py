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

from rrt_utils import sample_random_q, sample_random_object_pose, interpolate_state, find_nearest_neighbor

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
    parser = argparse.ArgumentParser(description="Visualize RRT Sampling and Interpolation")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the generated dataset file (parquet) or HF Hub ID')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host for Viser server')
    parser.add_argument('--port', type=int, default=8080, help='Port for Viser server')
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_grasp_dataset(args.dataset_path)
    print(f"Dataset loaded. Total grasps: {len(dataset)}")

    if len(dataset) == 0:
        print("Dataset is empty.")
        return

    # Pre-load dataset tensors for fast NN search
    # Ensure they are tensors
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
    # wxyz
    ROBOT_ROTATION = np.array([-0.707, 0, 0.707, 0])
    
    # Create root frames
    # We will have 3 sets: Nearest (Base), Random, Interpolated
    
    # Helper to create robot root
    def create_robot_root(name, visible=True):
        return server.scene.add_frame(
            name,
            wxyz=ROBOT_ROTATION,
            position=(0, 0, 0),
            show_axes=False,
            visible=visible
        )

    root_nearest = create_robot_root("/root_nearest", visible=True)
    root_rand = create_robot_root("/root_rand", visible=True)
    root_interp = create_robot_root("/root_interp", visible=True)

    # State variables
    current_robot_name = None
    current_mesh_path = None
    
    # URDF loaders
    urdf_nearest: Optional[ViserUrdf] = None
    urdf_rand: Optional[ViserUrdf] = None
    urdf_interp: Optional[ViserUrdf] = None
    
    # Object handles
    obj_nearest = None
    obj_rand = None
    obj_interp = None
    
    # Data
    lygra_robot = None
    tree = None
    urdf_model = None # yourdfpy model
    
    # RRT State
    state_nearest = None # (q, p, index)
    state_rand = None    # (q, p)
    state_interp = None  # (q, p)
    state_path = None    # (q_path, p_path)
    
    # GUI elements
    with server.gui.add_folder("RRT Controls"):
        sample_btn = server.gui.add_button("Sample New Random State")
        
        interp_slider = server.gui.add_slider(
            "Interpolation t",
            min=0.0,
            max=1.0,
            step=0.01,
            initial_value=0.0,
        )
        
    with server.gui.add_folder("Visualization Options"):
        show_nearest = server.gui.add_checkbox("Show Nearest (Base)", True)
        show_rand = server.gui.add_checkbox("Show Random Target", True)
        show_interp = server.gui.add_checkbox("Show Interpolated", True)
        
        show_contacts = server.gui.add_checkbox("Show Contacts", True)

    def load_robot_and_object(robot_name, mesh_path):
        nonlocal current_robot_name, current_mesh_path, lygra_robot, tree, urdf_model
        nonlocal urdf_nearest, urdf_rand, urdf_interp, obj_nearest, obj_rand, obj_interp
        
        if robot_name != current_robot_name:
            print(f"Loading robot: {robot_name}")
            lygra_robot = build_robot(robot_name)
            urdf_path = lygra_robot.urdf_path
            tree = build_kinematics_tree(urdf_path, lygra_robot.get_active_joints())
            urdf_model = yourdfpy.URDF.load(urdf_path)
            
            # Create ViserUrdfs
            # We can't easily change opacity of ViserUrdf, so we just use them as is.
            # Maybe we can use different root nodes to separate them.
            
            urdf_nearest = ViserUrdf(server, urdf_or_path=urdf_model, root_node_name="/root_nearest/robot")
            urdf_rand = ViserUrdf(server, urdf_or_path=urdf_model, root_node_name="/root_rand/robot")
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
                
                obj_nearest = create_obj("object", "/root_nearest")
                obj_rand = create_obj("object", "/root_rand")
                obj_interp = create_obj("object", "/root_interp")
                
                current_mesh_path = mesh_path
            else:
                print(f"Warning: Mesh file not found at {mesh_path}")

    def update_visualization():
        nonlocal state_nearest, state_rand, state_interp
        
        # Visibility
        root_nearest.visible = show_nearest.value
        root_rand.visible = show_rand.value
        root_interp.visible = show_interp.value
        
        # Helper to update pose
        def update_pose(urdf_viz, obj_handle, q, p):
            if urdf_viz and q is not None:
                urdf_viz.update_cfg(q.flatten().cpu().numpy())
            
            if obj_handle and p is not None:
                # p is T_robot_object
                # We need to apply T_world_robot to get T_world_object
                # But wait, our root frames (/root_nearest etc) already have T_world_robot applied!
                # So we just need to set object pose relative to root frame, which is exactly p!
                
                p_np = p.detach().cpu().numpy()[0] # [4, 4]
                R = p_np[:3, :3]
                t = p_np[:3, 3]
                
                rot = Rotation.from_matrix(R)
                quat = rot.as_quat() # xyzw
                wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
                
                obj_handle.position = t
                obj_handle.wxyz = wxyz

        if state_nearest:
            update_pose(urdf_nearest, obj_nearest, state_nearest[0], state_nearest[1])
            
        if state_rand:
            update_pose(urdf_rand, obj_rand, state_rand[0], state_rand[1])
            
        if state_interp:
            update_pose(urdf_interp, obj_interp, state_interp[0], state_interp[1])
            
            # Visualize Contacts for Interpolated State
            # We take contacts from Nearest and transform them
            if show_contacts.value and state_nearest and state_interp:
                idx = state_nearest[2]
                sample = dataset[int(idx)]
                
                if 'target_pos' in sample:
                    target_pos = sample['target_pos'] # [N, 3] in Robot Frame (at p_nearest)
                    if isinstance(target_pos, torch.Tensor):
                        target_pos = target_pos.detach().cpu().numpy()
                        
                    p_nearest = state_nearest[1].detach().cpu().numpy()[0] # [4, 4]
                    p_interp = state_interp[1].detach().cpu().numpy()[0]   # [4, 4]
                    
                    # Transform target_pos from Robot Frame (at p_nearest) to Object Frame
                    # T_robot_obj_nearest = p_nearest
                    # P_robot = T_robot_obj_nearest @ P_obj
                    # P_obj = inv(T_robot_obj_nearest) @ P_robot
                    
                    T_obj_robot_nearest = np.linalg.inv(p_nearest)
                    
                    target_pos_homog = np.concatenate([target_pos, np.ones((len(target_pos), 1))], axis=1) # [N, 4]
                    target_pos_obj = (T_obj_robot_nearest @ target_pos_homog.T).T # [N, 4]
                    
                    # Transform from Object Frame to Robot Frame (at p_interp)
                    # P_robot_interp = T_robot_obj_interp @ P_obj
                    target_pos_interp_robot = (p_interp @ target_pos_obj.T).T[:, :3]
                    
                    # Now visualize these points. They are in Robot Frame.
                    # But we want to visualize them in World Frame?
                    # Or relative to /root_interp?
                    # If we add them to /root_interp, they are in Robot Frame.
                    
                    server.scene.add_point_cloud(
                        name="/root_interp/target_pos",
                        points=target_pos_interp_robot,
                        colors=np.array([[1.0, 0.0, 0.0]] * len(target_pos_interp_robot)),
                        point_size=0.005
                    )
                    
                    # Also visualize contact_pos (on robot links)
                    # These move with the robot q automatically if we attach them to links?
                    # No, ViserUrdf doesn't let us attach points to links easily.
                    # We have to compute FK for interpolated q and place points.
                    
                    # Removed blue contact visualization as requested.
                    pass

    @sample_btn.on_click
    def _(_):
        nonlocal state_nearest, state_rand, state_interp, state_path
        
        # 1. Sample Random
        # We need robot instance to sample q
        # We assume dataset is homogeneous (same robot/object) for now, or we pick one.
        # Let's pick the first one to get robot/object info if not loaded.
        if lygra_robot is None:
            sample0 = dataset[0]
            load_robot_and_object(sample0['robot_name'], sample0['mesh_path'])
            
        q_rand = sample_random_q(lygra_robot)
        p_rand = sample_random_object_pose(lygra_robot)
        state_rand = (q_rand, p_rand)
        
        # 2. Find Nearest Neighbor
        idx = find_nearest_neighbor(q_rand, p_rand, dataset_q, dataset_p)
        
        # Load NN state
        sample_nn = dataset[int(idx)]
        q_nn = sample_nn['q'].unsqueeze(0)
        p_nn = sample_nn['object_pose'].unsqueeze(0)
        
        # Ensure robot/object matches (if dataset is mixed)
        if sample_nn['robot_name'] != current_robot_name or sample_nn['mesh_path'] != current_mesh_path:
            load_robot_and_object(sample_nn['robot_name'], sample_nn['mesh_path'])
            # Re-sample random if robot changed? 
            # For now assume single robot dataset.
            
        state_nearest = (q_nn, p_nn, idx)
        
        # 3. Interpolate (at current slider value)
        # t = interp_slider.value
        q_path, p_path = interpolate_state(q_nn, p_nn, q_rand, p_rand, num_steps=100)
        state_path = (q_path, p_path)
        
        idx_interp = int(interp_slider.value * (len(q_path) - 1))
        idx_interp = max(0, min(idx_interp, len(q_path) - 1))
        state_interp = (q_path[idx_interp:idx_interp+1], p_path[idx_interp:idx_interp+1])
        
        update_visualization()

    @interp_slider.on_update
    def _(_):
        nonlocal state_interp
        if state_path:
            q_path, p_path = state_path
            idx_interp = int(interp_slider.value * (len(q_path) - 1))
            idx_interp = max(0, min(idx_interp, len(q_path) - 1))
            state_interp = (q_path[idx_interp:idx_interp+1], p_path[idx_interp:idx_interp+1])
            update_visualization()
            
    @show_nearest.on_update
    def _(_): update_visualization()
    @show_rand.on_update
    def _(_): update_visualization()
    @show_interp.on_update
    def _(_): update_visualization()
    @show_contacts.on_update
    def _(_): update_visualization()

    # Initial load
    if len(dataset) > 0:
        sample0 = dataset[0]
        load_robot_and_object(sample0['robot_name'], sample0['mesh_path'])

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.3, 0.3, 0.3)
        client.camera.look_at = (0.0, 0.0, 0.0)

    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
