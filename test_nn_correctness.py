import torch
import numpy as np
from rrt_utils import find_nearest_neighbor
import math_utils

def test_nn():
    print("Generating mock dataset...")
    N = 100
    n_dof = 16
    dataset_q = torch.randn(N, n_dof)
    
    # Random poses
    dataset_p = []
    for _ in range(N):
        R = math_utils.generate_random_rotation()
        t = torch.randn(3)
        T = torch.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        dataset_p.append(T)
    dataset_p = torch.stack(dataset_p)
    
    print("Testing Self-Identification...")
    # 1. Check if querying with a dataset point returns itself
    # We pass the whole dataset as the query batch
    indices = find_nearest_neighbor(dataset_q, dataset_p, dataset_q, dataset_p)
    # Should be 0, 1, 2, ... N-1
    expected = torch.arange(N)
    mismatches = torch.sum(indices != expected)
    if mismatches == 0:
        print("Passed Self-Identification.")
    else:
        print(f"Self-identification failed. Mismatches: {mismatches}")
        # It's possible (though unlikely with floats) that two points are identical or equidistant.
    
    print("Testing Perturbation...")
    # 2. Perturb slightly
    q_perturbed = dataset_q + torch.randn_like(dataset_q) * 0.0001
    # For pose, just perturb translation for simplicity
    p_perturbed = dataset_p.clone()
    p_perturbed[:, :3, 3] += torch.randn(N, 3) * 0.0001
    
    indices = find_nearest_neighbor(q_perturbed, p_perturbed, dataset_q, dataset_p)
    mismatches = torch.sum(indices != expected)
    if mismatches == 0:
        print("Passed Perturbation.")
    else:
        print(f"Perturbation test failed. Mismatches: {mismatches}")
    
    print("Testing Brute Force Verification (Single Query)...")
    # 3. Random query vs Brute Force
    q_rand = torch.randn(1, n_dof)
    
    R = math_utils.generate_random_rotation()
    t = torch.randn(3)
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    p_rand = T.unsqueeze(0)
    
    idx = find_nearest_neighbor(q_rand, p_rand, dataset_q, dataset_p)
    
    # Manual calculation loop
    min_dist = float('inf')
    best_i = -1
    
    w_q = 1.0
    w_p_pos = 5.0
    w_p_rot = 1.0
    
    for i in range(N):
        d_q = torch.norm(dataset_q[i] - q_rand[0])
        
        t_data, rot_data = math_utils.unmake_pose(dataset_p[i].unsqueeze(0))
        q_data_quat = math_utils.quat_from_matrix(rot_data)
        
        t_rand_u, rot_rand_u = math_utils.unmake_pose(p_rand)
        q_rand_quat = math_utils.quat_from_matrix(rot_rand_u)
        
        pos_err, rot_err = math_utils.compute_pose_error(
            t_rand_u, q_rand_quat, t_data, q_data_quat, rot_error_type="axis_angle"
        )
        
        d_p_pos = torch.norm(pos_err)
        d_p_rot = torch.norm(rot_err)
        
        d_total = w_q * d_q + w_p_pos * d_p_pos + w_p_rot * d_p_rot
        
        if d_total < min_dist:
            min_dist = d_total
            best_i = i
            
    print(f"Vectorized Index: {idx.item()}, Brute Force Index: {best_i}")
    if idx.item() == best_i:
        print("Passed Brute Force Verification.")
    else:
        print("Brute Force Verification Failed!")

if __name__ == "__main__":
    test_nn()
