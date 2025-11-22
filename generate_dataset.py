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
from datasets import Dataset

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

def get_args():
    parser = argparse.ArgumentParser(description="Grasp Dataset Generation Script")
    parser.add_argument('--robot', type=str, default="leap", help='Robot Name')
    parser.add_argument('--batch_size_outer', type=int, default=128, help='Outer batch size (Object Pose)')
    parser.add_argument('--batch_size_inner', type=int, default=128, help='Inner batch size (Contact Domain Variants)')
    parser.add_argument('--n_batches', type=int, default=2, help='Number of batches to run')
    parser.add_argument('--n_grasps', type=int, default=-1, help='Total number of grasps to generate (overrides n_batches if > 0)')
    parser.add_argument('--n_contact', type=int, default=3, help='Number of non-static contacts to optimize')
    parser.add_argument('--n_sample_point', type=int, default=2048, help='Number of sampled object points')
    parser.add_argument('--ik_finetune_iter', type=int, default=5, help='Number of IK finetune iterations')
    parser.add_argument('--zo_lr_sigma', type=float, default=5, help='Sigma of the Zeroth-order Optimizer')
    parser.add_argument('--cf_accel', type=str, default='lbvhs2', help='Contact Field Acceleration Structure')
    parser.add_argument('--object_pose_sampling_strategy', type=str, default='canonical', help='Object pose sampling strategy')
    parser.add_argument('--object_mesh_path', type=str, default="./assets/40mm_cube.stl", help='Path to the object mesh')
    parser.add_argument('--output_dir', type=str, default="./outputs/grasp_dataset", help='Directory to save the dataset')
    parser.add_argument('--push_to_hub', type=str, default="iantc104/leap_hand_grasp_cube", help='Hugging Face Hub repository name to push to (e.g., "username/dataset")')

    args = parser.parse_args()
    return args

def generate_grasps(args, robot, tree, mesh_data, mesh_data_for_ik, decomposed_static_mesh_data, decomposed_mesh_data, 
                   self_collision_link_pairs, contact_field, dependency_sets, contact_parent_ids, dependency_matrix, 
                   accel_structure, object_mesh, points_all, normals_all, points, normals, gpu_memory_pool):
    
    batch_size_outer = args.batch_size_outer
    batch_size_inner = args.batch_size_inner
    n_contact = args.n_contact
    ik_finetune_iter = args.ik_finetune_iter
    object_pose_sampling_strategy = args.object_pose_sampling_strategy
    object_area = object_mesh.get_area()
    zo_lr = ((object_area / args.n_sample_point) ** 0.5) * args.zo_lr_sigma

    with torch.no_grad():
        # Object Placement
        object_poses, condition = sample_object_pose(
            n=batch_size_outer, 
            points=points, 
            normals=normals, 
            contact_field=contact_field, 
            tree=tree, 
            mesh_data=decomposed_static_mesh_data,
            sampling_args=get_object_pose_sampling_args(object_pose_sampling_strategy, robot)
        )

        # Contact Field BVH Traversal
        interaction_matrix_hand_point_idx = batch_object_all_contact_fields_interaction(
            object_pos=points, 
            object_normal=normals, 
            object_pose=object_poses, 
            accel_structure=accel_structure
        )

        interaction_matrix = (interaction_matrix_hand_point_idx >= 0).int()
        link_interaction_matrix = contact_field.reduce_link_interaction(interaction_matrix)

        # Get Contact Domain
        contact_domain_pos, contact_domain_normal, contact_domain_point_idx, \
        object_poses, contact_link_ids, condition, valid_outer_idx = \
        sample_pose_and_contact_from_interaction(
            n_contact=n_contact,
            interaction_matrix=link_interaction_matrix, 
            dependency_matrix=dependency_matrix, 
            object_points=points, 
            object_normals=normals, 
            object_poses=object_poses,
            condition=condition
        )

        # Search Contact Points in Contact Domain
        target_contact_pos, target_contact_normal, target_contact_point_idx, \
        object_poses, target_contact_link_ids, target_batch_outer_ids = \
        search_contact_point(
            contact_domain_pos=contact_domain_pos, 
            contact_domain_normal=contact_domain_normal, 
            contact_domain_point_idx=contact_domain_point_idx,
            object_poses=object_poses, 
            contact_ids=contact_link_ids,
            batch_size=batch_size_inner,
            return_hand_frame=True,
            condition=condition,
            zo_lr=zo_lr
        )

        contact_ids, local_contact_ids = contact_field.sample_contact_ids(
            interaction_matrix=interaction_matrix[valid_outer_idx], 
            interaction_matrix_hand_point_idx=interaction_matrix_hand_point_idx[valid_outer_idx],
            target_batch_outer_ids=target_batch_outer_ids, 
            target_contact_link_ids=target_contact_link_ids, 
            target_contact_point_idx=target_contact_point_idx
        )

        contact_pos_in_linkf, contact_normal_in_linkf = contact_field.sample_contact_geometry(contact_ids, local_contact_ids)

        if len(contact_ids) == 0:
            return {}

        # Kinematics Optimization (I)
        result = batch_ik(
            tree=tree,
            contact_ids=contact_ids,
            contact_parent_ids=contact_parent_ids,
            contact_pos_in_linkf=contact_pos_in_linkf.float(),
            contact_normal_in_linkf=contact_normal_in_linkf.float(),
            target_contact_pos=target_contact_pos.float(),
            target_contact_normal=target_contact_normal.float(),
            object_pose=object_poses.float(),
            gpu_memory_pool=gpu_memory_pool
        )
        
        # Kinematics Optimization (II)
        result = batch_contact_adjustment(
            tree=tree,
            mesh=mesh_data_for_ik,
            q_init=result["q"],
            q_mask=result["q_mask"],
            contact_ids=contact_ids,
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

        # Postprocessing
        result = batch_assign_free_finger_and_filter(
            tree=tree,
            result=result,
            object_point=points_all,
            self_collision_link_pairs=self_collision_link_pairs,
            decomposed_mesh_data=decomposed_mesh_data
        )

    return result, contact_ids

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
        max_batch=min([args.batch_size_outer * args.batch_size_inner, 65536]), 
        retry=10
    )

    # -----------------
    # Generation Loop
    # -----------------
    all_results = []
    
    if args.n_grasps > 0:
        total_grasps_needed = args.n_grasps
        print(f"Starting grasp generation for {total_grasps_needed} grasps...")
    else:
        total_grasps_needed = None
        print(f"Starting grasp generation for {args.n_batches} batches...")
    
    batch_count = 0
    with tqdm() as pbar:
        while True:
            if total_grasps_needed is not None and len(all_results) >= total_grasps_needed:
                break
            if total_grasps_needed is None and batch_count >= args.n_batches:
                break
            
            result, contact_ids = generate_grasps(
                args, robot, tree, mesh_data, mesh_data_for_ik, decomposed_static_mesh_data, decomposed_mesh_data,
                self_collision_link_pairs, contact_field, dependency_sets, contact_parent_ids, dependency_matrix,
                accel_structure, object_mesh, points_all, normals_all, points, normals, gpu_memory_pool
            )
            
            # Convert tensors to numpy for storage
            batch_data = {}
            n_items = 0
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    batch_data[k] = v.cpu().numpy()
                    n_items = len(batch_data[k])
                else:
                    # Handle non-tensor data if any, though result usually contains tensors
                    pass
            
            if n_items > 0:
                # Convert dict of arrays to list of dicts
                keys = list(batch_data.keys())
                for idx in range(n_items):
                    if total_grasps_needed is not None and len(all_results) >= total_grasps_needed:
                        break
                    item = {key: batch_data[key][idx] for key in keys}
                    item['contact_ids'] = contact_ids.cpu().numpy()
                    item['batch_index'] = batch_count
                    item['mesh_path'] = args.object_mesh_path
                    item['robot_name'] = args.robot
                    all_results.append(item)
            
            batch_count += 1
            pbar.update(1)
            if total_grasps_needed is None:
                pbar.set_description(f"Batch {batch_count}/{args.n_batches}")
            else:
                pbar.set_description(f"Batch {batch_count} (total grasps: {len(all_results)}/{total_grasps_needed})")

    print(f"Generated {len(all_results)} valid grasps.")

    # -----------------
    # Save Dataset
    # -----------------
    if len(all_results) > 0:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Convert numpy arrays to lists for compatibility
        for item in all_results:
            for k, v in item.items():
                if isinstance(v, np.ndarray):
                    item[k] = v.tolist()

        # Create Hugging Face Dataset directly from list of dicts
        ds = Dataset.from_list(all_results)
        
        # Save to disk (Arrow format) which is native to datasets
        # This preserves types better than parquet sometimes, but parquet is also fine.
        # Let's save as parquet for portability as requested before, but also support push_to_hub
        
        output_file = os.path.join(args.output_dir, f"grasps_{args.robot}.parquet")
        print(f"Saving dataset to {output_file}...")
        ds.to_parquet(output_file)
        
        if args.push_to_hub:
            print(f"Pushing dataset to Hugging Face Hub: {args.push_to_hub}...")
            ds.push_to_hub(args.push_to_hub)
            
        print("Done.")
    else:
        print("No valid grasps found.")

if __name__ == '__main__':
    args = get_args()
    main(args)
