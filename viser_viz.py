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
except ImportError:
    print("Lygra package not found. Please ensure you are in the correct environment.")
    exit(1)

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
    parser = argparse.ArgumentParser(description="Visualize grasps with Viser")
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

    # Start Viser server
    server = viser.ViserServer(host=args.host, port=args.port)
    print(f"Viser server started at http://{args.host}:{args.port}")

    # Robot orientation
    # wxyz
    ROBOT_ROTATION = np.array([-0.707, 0, 0.707, 0])
    
    # Create a root frame for the robot to apply the rotation
    robot_root = server.scene.add_frame(
        "/robot_root",
        wxyz=ROBOT_ROTATION,
        position=(0, 0, 0),
        show_axes=False
    )

    # State variables
    current_robot_name = None
    current_mesh_path = None
    viser_urdf: Optional[ViserUrdf] = None
    object_handle = None
    
    # GUI elements
    with server.gui.add_folder("Grasp Control"):
        grasp_index_slider = server.gui.add_slider(
            "Grasp Index",
            min=0,
            max=len(dataset) - 1,
            step=1,
            initial_value=0,
        )
        
        prev_btn = server.gui.add_button("Previous")
        next_btn = server.gui.add_button("Next")
            
        play_btn = server.gui.add_button("Play")
        stop_btn = server.gui.add_button("Stop")
        stop_btn.visible = False

    with server.gui.add_folder("Visualization Options"):
        show_robot_mesh = server.gui.add_checkbox("Show Robot Mesh", True)
        show_object_mesh = server.gui.add_checkbox("Show Object Mesh", True)
        show_grid = server.gui.add_checkbox("Show Grid", True)

    # Grid
    grid = server.scene.add_grid("grid", width=1, height=1, cell_size=0.1)
    
    @show_grid.on_update
    def _(_):
        grid.visible = show_grid.value

    def update_scene(index):
        nonlocal current_robot_name, current_mesh_path, viser_urdf, object_handle
        
        # Get data sample
        sample = dataset[int(index)]
        
        robot_name = sample['robot_name']
        mesh_path = sample['mesh_path']
        q = sample['q']
        object_pose = sample['object_pose']
        
        # Convert to numpy
        if isinstance(q, torch.Tensor):
            q = q.detach().cpu().numpy()
        if isinstance(object_pose, torch.Tensor):
            object_pose = object_pose.detach().cpu().numpy()
            
        # Check if robot changed
        if robot_name != current_robot_name:
            print(f"Loading robot: {robot_name}")
            # Note: We don't explicitly remove the old robot here because ViserUrdf doesn't expose a clean remove.
            # Ideally we would track the root node and remove it.
            # Assuming ViserUrdf creates a node named after the robot or we can find it.
            # For now, we just create a new one. If this causes overlap, we might need to clear scene.
            
            # Build robot to get URDF path
            lygra_robot = build_robot(robot_name)
            urdf_path = lygra_robot.urdf_path
            
            # Load URDF
            urdf = yourdfpy.URDF.load(urdf_path)

            viser_urdf = ViserUrdf(
                server,
                urdf_or_path=urdf,
                load_meshes=True,
                load_collision_meshes=False,
                root_node_name="/robot_root"
            )
            current_robot_name = robot_name
            
            # Bind visibility
            viser_urdf.show_visual = show_robot_mesh.value
            
        # Check if object changed
        if mesh_path != current_mesh_path:
            print(f"Loading object: {mesh_path}")
            if object_handle is not None:
                object_handle.remove()
            
            if os.path.exists(mesh_path):
                mesh_obj = MeshObject(mesh_path)
                # Viser expects trimesh
                object_handle = server.scene.add_mesh_trimesh(
                    name="/object",
                    mesh=mesh_obj.mesh,
                    position=(0,0,0),
                    wxyz=(1,0,0,0) # Identity quaternion
                )
                current_mesh_path = mesh_path
                object_handle.visible = show_object_mesh.value
            else:
                print(f"Warning: Mesh file not found at {mesh_path}")
                object_handle = None

        # Update Robot Configuration
        if viser_urdf:
            viser_urdf.update_cfg(q)

        # Update Object Pose
        if object_handle:
            # object_pose is 4x4 matrix (T_robot_object)
            T_robot_object = object_pose
            
            # Construct T_world_robot
            T_world_robot = np.eye(4)
            # Rotation from quaternion (wxyz) -> matrix
            # scipy uses xyzw
            r = Rotation.from_quat([ROBOT_ROTATION[1], ROBOT_ROTATION[2], ROBOT_ROTATION[3], ROBOT_ROTATION[0]])
            T_world_robot[:3, :3] = r.as_matrix()
            
            # T_world_object = T_world_robot @ T_robot_object
            T_world_object = T_world_robot @ T_robot_object
            
            R = T_world_object[:3, :3]
            t = T_world_object[:3, 3]
            
            rot = Rotation.from_matrix(R)
            quat = rot.as_quat() # xyzw
            # Viser wants wxyz
            wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])

            object_handle.position = t
            object_handle.wxyz = wxyz

    # Callbacks
    @grasp_index_slider.on_update
    def _(_):
        update_scene(grasp_index_slider.value)

    @prev_btn.on_click
    def _(_):
        val = grasp_index_slider.value - 1
        if val >= 0:
            grasp_index_slider.value = val

    @next_btn.on_click
    def _(_):
        val = grasp_index_slider.value + 1
        if val < len(dataset):
            grasp_index_slider.value = val
            
    @show_robot_mesh.on_update
    def _(_):
        if viser_urdf:
            viser_urdf.show_visual = show_robot_mesh.value

    @show_object_mesh.on_update
    def _(_):
        if object_handle:
            object_handle.visible = show_object_mesh.value

    # Playback logic
    playing = False
    def playback_loop():
        nonlocal playing
        while True:
            if playing:
                val = grasp_index_slider.value + 1
                if val < len(dataset):
                    grasp_index_slider.value = val
                else:
                    playing = False
                    play_btn.visible = True
                    stop_btn.visible = False
                time.sleep(0.5) # Adjust speed
            else:
                time.sleep(0.1)

    import threading
    threading.Thread(target=playback_loop, daemon=True).start()

    @play_btn.on_click
    def _(_):
        nonlocal playing
        playing = True
        play_btn.visible = False
        stop_btn.visible = True

    @stop_btn.on_click
    def _(_):
        nonlocal playing
        playing = False
        play_btn.visible = True
        stop_btn.visible = False

    # Initial update
    update_scene(grasp_index_slider.value)
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.3, 0.3, 0.3)
        client.camera.look_at = (0.0, 0.0, 0.0)

    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
