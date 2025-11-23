# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets

# Lygra Common
from lygra.robot import build_robot
from lygra.contact_set import get_dependency_matrix, get_link_dependency_matrix
from lygra.kinematics import build_kinematics_tree
from lygra.mesh import get_urdf_mesh, get_urdf_mesh_decomposed, get_urdf_mesh_for_projection
from lygra.mesh_analyzer import get_support_point_mask
from lygra.utils.geom_utils import MeshObject
from lygra.memory import IKGPUBufferPool
from lygra.pipeline.module.object_placement import sample_object_pose, get_object_pose_sampling_args
from lygra.pipeline.module.contact_query import batch_object_all_contact_fields_interaction
from lygra.pipeline.module.contact_collection import sample_pose_and_contact_from_interaction
from lygra.pipeline.module.contact_optimization import search_contact_point
from lygra.pipeline.module.kinematics import batch_ik, batch_contact_adjustment
from lygra.pipeline.module.postprocess import batch_assign_free_finger_and_filter

from rrt_utils import (
    sample_random_q,
    sample_random_object_pose,
    interpolate_state_tau,
    find_nearest_neighbor,
)
import math_utils

def get_args():
    parser = argparse.ArgumentParser(description="Grasp Dataset Generation Script")
    parser.add_argument('--dataset_path', type=str, default="iantc104/leap_hand_grasp_cube", help='Hugging Face Hub dataset path')
    parser.add_argument('--robot', type=str, default="leap", help='Robot Name')
    parser.add_argument('--batch_size', type=int, default=4096, help='Outer batch size (Object Pose)')
    parser.add_argument('--batch_cutoff', type=int, default=1000, help='Batch size cutoff for processing')
    parser.add_argument('--n_batches', type=int, default=2, help='Number of batches to run')
    parser.add_argument('--n_grasps', type=int, default=200000, help='Total number of grasps to generate (overrides n_batches if > 0)')
    parser.add_argument('--n_contact', type=int, default=3, help='Number of non-static contacts to optimize')
    parser.add_argument('--n_sample_point', type=int, default=2048, help='Number of sampled object points')
    parser.add_argument('--ik_finetune_iter', type=int, default=5, help='Number of IK finetune iterations')
    parser.add_argument('--zo_lr_sigma', type=float, default=5, help='Sigma of the Zeroth-order Optimizer')
    parser.add_argument('--cf_accel', type=str, default='lbvhs2', help='Contact Field Acceleration Structure')
    parser.add_argument('--object_pose_sampling_strategy', type=str, default='canonical', help='Object pose sampling strategy')
    parser.add_argument('--object_mesh_path', type=str, default="./assets/40mm_cube.stl", help='Path to the object mesh')
    parser.add_argument('--output_dir', type=str, default="./outputs/leap_hand_grasp_cube_rrt", help='Directory to save the dataset')
    parser.add_argument('--push_to_hub', type=str, default="iantc104/leap_hand_grasp_cube_rrt", help='Hugging Face Hub repository name to push to (e.g., "username/dataset")')
    parser.add_argument('--nn_downsample_size', type=int, default=2048, help='Number of samples to use for nearest neighbor search')
    parser.add_argument('--rrt_tau', type=float, default=0.2, help='RRT interpolation step size (tau)')
    args = parser.parse_args()
    return args

# cycle daterloader iterator
def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def generate_grasps(
    args, robot, tree, mesh_data_for_ik, decomposed_mesh_data, 
    self_collision_link_pairs, points_all, gpu_memory_pool, nn_dataset
):
    # Pre-load dataset to GPU
    dataset_q = torch.stack(list(nn_dataset['q'])).cuda() # (N, n_dof)
    dataset_p = torch.stack(list(nn_dataset['object_pose'])).cuda() # (N, 4, 4)
    dataset_target_pos = torch.stack(list(nn_dataset['target_pos'])).cuda() # (N, n_contact, 3)
    dataset_target_normal = torch.stack(list(nn_dataset['target_normal'])).cuda() # (N, n_contact, 3)
    dataset_contact_pos = torch.stack(list(nn_dataset['contact_pos'])).cuda() # (N, n_contact, 3)
    dataset_contact_normal = torch.stack(list(nn_dataset['contact_normal'])).cuda() # (N, n_contact, 3)
    dataset_contact_link_id = torch.stack(list(nn_dataset['contact_link_id'])).cuda() # (N, n_contact)

    ik_finetune_iter = args.ik_finetune_iter

    with torch.no_grad():
        # 1. Sample Random
        q_rand_batch = sample_random_q(robot, batch_size=args.batch_size).cuda() # (B, n_dof)
        p_rand_batch = sample_random_object_pose(robot, batch_size=args.batch_size).cuda() # (B, 4, 4)

        # 2. Find Nearest Neighbor (Batched)
        print("Finding nearest neighbors...")
        indices = find_nearest_neighbor(q_rand_batch, p_rand_batch, dataset_q, dataset_p) # (B,)
        
        # Gather NN data
        q_nn_batch = dataset_q[indices] # (B, n_dof)
        p_nn_batch = dataset_p[indices] # (B, 4, 4)
        target_pos_batch = dataset_target_pos[indices] # (B, n_contact, 3)
        target_normal_batch = dataset_target_normal[indices] # (B, n_contact, 3)
        contact_pos_in_linkf = dataset_contact_pos[indices] # (B, n_contact, 3)
        contact_normal_in_linkf = dataset_contact_normal[indices] # (B, n_contact, 3)
        contact_link_ids = dataset_contact_link_id[indices] # (B, n_contact)

        # 3. Interpolate (Batched)
        q_interp, p_interp = interpolate_state_tau(q_nn_batch, p_nn_batch, q_rand_batch, p_rand_batch, tau=args.rrt_tau)

        # 4. Adjust target contact positions
        # target_pos_batch: [B, n_contact, 3]
        B, n_contact, _ = target_pos_batch.shape
        
        target_pos_homog = torch.cat([target_pos_batch, torch.ones((B, n_contact, 1)).cuda()], dim=2) # [B, n_contact, 4]
        
        p_nn_inv = torch.linalg.pinv(p_nn_batch) # [B, 4, 4]
        # [B, 1, 4, 4] @ [B, n_contact, 4, 1] -> [B, n_contact, 4, 1]
        # Debug shapes
        # print(f"p_nn_inv: {p_nn_inv.shape}")
        # print(f"target_pos_homog: {target_pos_homog.shape}")
        
        target_pos_obj = torch.matmul(p_nn_inv.unsqueeze(1), target_pos_homog.unsqueeze(-1)).squeeze(-1)
        
        # print(f"target_pos_obj: {target_pos_obj.shape}")
        # print(f"p_interp: {p_interp.shape}")

        # [B, 1, 4, 4] @ [B, n_contact, 4, 1] -> [B, n_contact, 4, 1]
        # Ensure target_pos_obj is unsqueezed for matmul
        target_pos_interp_homog = torch.matmul(p_interp.unsqueeze(1), target_pos_obj.unsqueeze(-1)).squeeze(-1)
        target_contact_pos = target_pos_interp_homog[:, :, :3] # [B, n_contact, 3]

        # 5. Adjust target normals
        # N_interp = R_interp @ R_nearest^T @ N_nearest
        R_nn = p_nn_batch[:, :3, :3] # [B, 3, 3]
        R_interp = p_interp[:, :3, :3] # [B, 3, 3]
        
        R_nn_T = R_nn.transpose(1, 2) # [B, 3, 3]
        R_rel = torch.matmul(R_interp, R_nn_T) # [B, 3, 3]
        
        # [B, 1, 3, 3] @ [B, n_contact, 3, 1] -> [B, n_contact, 3, 1]
        target_contact_normal = torch.matmul(R_rel.unsqueeze(1), target_normal_batch.unsqueeze(-1)).squeeze(-1) # [B, n_contact, 3]

        # Assign to variables for IK
        q = q_interp
        object_poses = p_interp

        print(f"Starting with {q.shape[0]} grasp candidates from RRT expansion...")

        # Kinematics Optimization (I)
        result = batch_ik(
            tree=tree,
            contact_link_ids=contact_link_ids,
            contact_pos_in_linkf=contact_pos_in_linkf.float(),
            contact_normal_in_linkf=contact_normal_in_linkf.float(),
            target_contact_pos=target_contact_pos.float(),
            target_contact_normal=target_contact_normal.float(),
            object_pose=object_poses.float(),
            gpu_memory_pool=gpu_memory_pool,
            q_init=q,
        )

        print(f"After IK, {result['q'].shape[0]} grasp candidates remain.")
        
        # Kinematics Optimization (II)
        result = batch_contact_adjustment(
            tree=tree,
            mesh=mesh_data_for_ik,
            q_init=result["q"],
            q_mask=result["q_mask"],
            contact_link_ids=result["contact_link_id"],
            contact_pos_in_linkf=result["contact_pos"],
            contact_normal_in_linkf=result["contact_normal"],
            target_contact_pos=result["target_pos"],
            target_contact_normal=result["target_normal"],
            object_pose=result["object_pose"],
            n_iter=ik_finetune_iter,
            gpu_memory_pool=gpu_memory_pool,
            ret_mesh_buffer=True
        )

        print(f"After IK Finetune, {result['q'].shape[0]} grasp candidates remain.")

        # Postprocessing
        result = batch_assign_free_finger_and_filter(
            tree=tree,
            result=result,
            object_point=points_all,
            self_collision_link_pairs=self_collision_link_pairs,
            decomposed_mesh_data=decomposed_mesh_data
        )

        print(f"After Collision Checking and Free Finger Assignment, {result['q'].shape[0]} grasps remain.")

    return result

def main(args):
    # -----------------
    # Preparation Stage 
    # -----------------
    robot = build_robot(args.robot)

    # Robot Structure.
    tree = build_kinematics_tree(
        urdf_path=robot.urdf_path,
        active_joint_names=robot.get_active_joints()
    )

    # Robot Mesh Data
    mesh_data = get_urdf_mesh(
        urdf_path=robot.urdf_path,
        tree=tree,
        mesh_scale=robot.get_mesh_scale()
    )

    mesh_data_for_ik = get_urdf_mesh_for_projection(
        urdf_path=robot.urdf_path,
        tree=tree,
        config=robot.get_contact_field_config(),
        mesh_scale=robot.get_mesh_scale()
    )

    decomposed_static_mesh_data = get_urdf_mesh_decomposed(
        urdf_path=robot.urdf_path,
        tree=tree,
        override_link_names=robot.get_static_links(),
        mesh_scale=robot.get_mesh_scale()
    )

    decomposed_mesh_data = get_urdf_mesh_decomposed(
        urdf_path=robot.urdf_path,
        tree=tree,
        mesh_scale=robot.get_mesh_scale()
    )

    # Robot Collision & Kinematics Metadata
    self_collision_link_pairs = tree.get_self_collision_check_link_pairs(
        link_body_id=decomposed_mesh_data['link_body_id'],
        whitelist_link=[]
    )

    self_collision_link_pairs = torch.from_numpy(self_collision_link_pairs).cuda().int()

    contact_field = robot.get_contact_field()
    dependency_sets = tree.get_dependency_sets([robot.get_base_link()])

    contact_parent_links = contact_field.get_all_parent_link_names()
    contact_parent_ids = [tree.get_link_id(link) for link in contact_parent_links]
    contact_parent_ids = torch.tensor(contact_parent_ids).cuda()

    dependency_matrix = get_link_dependency_matrix(contact_field, dependency_sets)
    dependency_matrix = dependency_matrix.cuda()

    # Contact Field Acceleration Data Structure (LBVH-S2Bundle)
    accel_structure = contact_field.generate_acceleration_structure(method=args.cf_accel)

    # Object Data.
    object_mesh = MeshObject(args.object_mesh_path)
    points, normals = object_mesh.sample_point_and_normal(count=args.n_sample_point)
    points_all = torch.from_numpy(points).cuda().float()
    normals_all = torch.from_numpy(normals).cuda().float()

    # Filtering
    support_point_mask = get_support_point_mask(points_all, normals_all, [0.01])[0]
    points = points_all[torch.where(support_point_mask)]            # good grasp point.
    normals = normals_all[torch.where(support_point_mask)]          # good_grasp_point.

    # IK GPU buffer. 
    gpu_memory_pool = IKGPUBufferPool(
        n_dof=tree.n_dof(), 
        n_link=tree.n_link(), 
        max_batch=min([16384, 65536]), 
        retry=10
    )

    # dataset to sample nearest neighbor from
    dataset = load_dataset(args.dataset_path, split="train")
    dataset = dataset.with_format("torch")
    dataset_keys = dataset.column_names

    # -----------------
    # Generation Loop
    # -----------------
    
    if args.n_grasps > 0:
        total_grasps_needed = args.n_grasps
        print(f"Starting grasp generation for {total_grasps_needed} grasps...")
    else:
        total_grasps_needed = None
        print(f"Starting grasp generation for {args.n_batches} batches...")
    
    batch_count = 0
    total_grasps_generated = len(dataset)

    with tqdm(total=total_grasps_needed if total_grasps_needed else args.n_batches, unit="grasps" if total_grasps_needed else "batches") as pbar:
        while True:
            if total_grasps_needed is not None and total_grasps_generated >= total_grasps_needed:
                break
            if total_grasps_needed is None and batch_count >= args.n_batches:
                break


            # select args.nn_downsample_size samples from dataset
            dataset_indices = np.random.choice(len(dataset), size=min(args.nn_downsample_size, len(dataset)), replace=False)
            nn_dataset = dataset.select(dataset_indices)
            
            result = generate_grasps(
                args, robot, tree, mesh_data_for_ik, decomposed_mesh_data, 
                self_collision_link_pairs, points_all, gpu_memory_pool, nn_dataset
            )

            # Convert tensors to numpy for storage
            batch_size = result['q'].shape[0] if 'q' in result else 0

            if batch_size >= 0:
                n_grasps = min(args.batch_cutoff, batch_size) if args.batch_cutoff > 0 else batch_size
                # for each dataset.column_names, create HF dataset from result
                result_np = {k:v[:n_grasps].cpu().numpy() for k,v in result.items() if k in dataset_keys}
                result_dataset = Dataset.from_dict(result_np)
                # concatenate to dataset
                dataset = concatenate_datasets([dataset, result_dataset])
                dataset = dataset.with_format("torch")
                batch_count += 1
                total_grasps_generated += n_grasps
                
                if total_grasps_needed:
                    pbar.update(n_grasps)
                else:
                    pbar.update(1)
            else:
                print("No valid grasps generated in this batch, skipping dataset update.")
            
            if total_grasps_needed is None:
                pbar.set_description(f"Batch {batch_count}/{args.n_batches}")
            else:
                pbar.set_description(f"Batch {batch_count} (total grasps: {total_grasps_generated}/{total_grasps_needed})")

    print(f"Generated {total_grasps_generated} valid grasps.")

    # -----------------
    # Save Dataset
    # -----------------
    if total_grasps_generated > 0:
        # Truncate if needed (assuming we want to add exactly n_grasps new grasps)
        # But wait, dataset contains (Original + New).
        # If we want to truncate the *new* grasps to exactly total_grasps_needed.
        # We need to know the original size.
        # However, the user request is "dataset will contain exactly n_grasps amount".
        # This is ambiguous. But given generate_dataset behavior (total size = n_grasps),
        # and grasp_rrt_expand behavior (append), maybe they want the *added* amount to be exactly n_grasps?
        # Or maybe they want the final file to have exactly n_grasps?
        # If I run expand with n_grasps=1000, I probably want 1000 new grasps.
        # So I should truncate the dataset to (original_size + n_grasps).
        
        # But I don't have original_size easily available here without re-reading or storing it.
        # Actually I can just use len(dataset) - total_grasps_generated to get original size?
        # No, total_grasps_generated tracks what we added.
        
        if total_grasps_needed is not None and total_grasps_generated > total_grasps_needed:
             print(f"Truncating generated grasps from {total_grasps_generated} to {total_grasps_needed}.")
             # We want to keep (len(dataset) - total_grasps_generated) + total_grasps_needed
             original_size = len(dataset) - total_grasps_generated
             final_size = original_size + total_grasps_needed
             # Dataset slicing in HF datasets
             dataset = dataset.select(range(final_size))
             total_grasps_generated = total_grasps_needed

        os.makedirs(args.output_dir, exist_ok=True)
            
        output_file_grasps = os.path.join(args.output_dir, f"grasps_rrt_{args.robot}.parquet")
        print(f"Saving grasps dataset to {output_file_grasps}...")
        dataset.to_parquet(output_file_grasps)
        
        if args.push_to_hub:
            print(f"Pushing dataset to Hugging Face Hub: {args.push_to_hub}...")
            # Push as separate configurations to allow different schemas
            dataset.push_to_hub(args.push_to_hub, split="train")
            
        print("Done.")
    else:
        print("No valid grasps found.")

if __name__ == '__main__':
    args = get_args()
    main(args)
