import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import open3d as o3d
from datasets import load_dataset

# Lygra imports for visualization
try:
    from lygra.robot import build_robot
    from lygra.utils.robot_visualizer import RobotVisualizer
    from lygra.utils.geom_utils import MeshObject
    from lygra.mesh import trimesh_to_open3d
except ImportError:
    print("Lygra package not found. Visualization will not work.")

def get_dataloader(data_path, batch_size=32, shuffle=True):
    """
    Creates a PyTorch DataLoader from a parquet file using Hugging Face datasets.
    """
    if data_path.endswith('.parquet'):
        dataset = load_dataset("parquet", data_files=data_path, split="train")
    else:
        # Assume it's a directory or hub path
        try:
            dataset = load_dataset(data_path, split="train")
        except:
             # Fallback for local directory created by save_to_disk (if we used that)
             from datasets import load_from_disk
             dataset = load_from_disk(data_path)

    # Set format to torch to automatically convert arrays to tensors
    dataset = dataset.with_format("torch")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def visualize_grasp(viewer, q, object_pose, object_mesh_original):
    # Get robot mesh
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()
    if isinstance(object_pose, torch.Tensor):
        object_pose = object_pose.detach().cpu().numpy()

    # RobotVisualizer.get_mesh_fk expects numpy array or tensor.
    robot_meshes = viewer.get_mesh_fk(q, mode='o3d', visual=True)
    
    # Get object mesh
    obj_mesh = object_mesh_original.mesh.copy()
    obj_mesh.apply_transform(object_pose)
    obj_mesh_o3d = trimesh_to_open3d(obj_mesh)
    obj_mesh_o3d.compute_vertex_normals()
    
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLitTransparency"
    material.base_color = [245 / 255, 162 / 255, 98 / 255, 0.8]
    material.base_metallic = 0.0
    material.base_roughness = 1.0
    
    object_mesh_dict = {"name": 'object', "geometry": obj_mesh_o3d, "material": material}
    
    viewer.show(robot_meshes + [object_mesh_dict])

if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the generated dataset file (parquet) or HF Hub ID')
    parser.add_argument('--visualize', action='store_true', help='Visualize loaded grasps')
    args = parser.parse_args()

    dataloader = get_dataloader(args.dataset_path, batch_size=4, shuffle=True)
    
    # Get dataset from dataloader to access metadata if needed (a bit hacky with DataLoader)
    # Better to just peek at the first batch
    
    print(f"Dataset loaded. Batches: {len(dataloader)}")

    viewer = None
    object_mesh_original = None

    for i, batch in enumerate(dataloader):
        print(f"Batch {i}: keys={batch.keys()}")
        print("q shape:", batch['q'].shape)
        print("object_pose shape:", batch['object_pose'].shape)
        
        if args.visualize:
            if viewer is None:
                # Initialize viewer on first batch
                # We need to get metadata. Since we are using standard DataLoader, 
                # we might have lost the dataset metadata access if we didn't preserve it.
                # But 'robot_name' and 'mesh_path' are columns in the batch!
                
                robot_name = batch['robot_name'][0]
                mesh_path = batch['mesh_path'][0]
                
                print(f"Initializing visualizer for robot: {robot_name}, object: {mesh_path}")
                robot = build_robot(robot_name)
                viewer = RobotVisualizer(robot)
                object_mesh_original = MeshObject(mesh_path)

            # Visualize first grasp in batch
            print("Visualizing first grasp in batch...")
            q = batch['q'][0]
            object_pose = batch['object_pose'][0]
            visualize_grasp(viewer, q, object_pose, object_mesh_original)
            
            if input("Continue? (Y/n) ") in ['n', 'N']:
                break
            
        # Break after a few batches for demo if not visualizing
        if not args.visualize and i >= 2:
            break
