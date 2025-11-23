import torch
import numpy as np
import math_utils
import math

def sample_random_q(robot, batch_size=1):
    """
    Sample random joint configurations within limits.
    """
    tree = robot.get_kinematics_tree()
    lowers = np.array(tree.joint_limit_lowers)
    uppers = np.array(tree.joint_limit_uppers)
    
    # Filter for active joints
    active_joint_ids = [tree.get_joint_id(name) for name in robot.get_active_joints()]
    active_lowers = torch.tensor(lowers[active_joint_ids], dtype=torch.float32)
    active_uppers = torch.tensor(uppers[active_joint_ids], dtype=torch.float32)
    
    # Sample
    rand = torch.rand((batch_size, len(active_joint_ids)))
    q_rand = active_lowers + rand * (active_uppers - active_lowers)
    return q_rand

def sample_random_object_pose(robot, batch_size=1):
    """
    Sample random object poses (position + orientation).
    Position in canonical space.
    """
    bmin, bmax = robot.get_canonical_space()
    bmin_t = torch.tensor(bmin, dtype=torch.float32)
    bmax_t = torch.tensor(bmax, dtype=torch.float32)
    
    poses = []
    for _ in range(batch_size):
        # Rotation
        R = math_utils.generate_random_rotation()
        
        # Translation
        pos = bmin_t + torch.rand(3) * (bmax_t - bmin_t)
        
        T = torch.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pos
        poses.append(T)
    
    return torch.stack(poses)

def interpolate_state(q_start, p_start, q_end, p_end, num_steps=10):
    """
    Interpolate between two states.
    Returns path of states.
    NO BATCHING.
    """
    assert q_start.ndim == 1 # [n_dof]
    assert q_end.ndim == 1 # [n_dof]
    assert p_start.ndim == 2 
    assert p_end.ndim == 2

    p_path, n = math_utils.interpolate_poses(p_start, p_end, num_steps=num_steps) # [n+2, 4, 4], n
        
    t = torch.linspace(0, 1, steps=n+2, device=q_start.device) # [n+2]
    q_path = q_start.unsqueeze(0) + (q_end - q_start).unsqueeze(0) * t.unsqueeze(1) # [n+2, n_dof]
    
    return q_path, p_path

def interpolate_state_tau(q_start, p_start, q_end, p_end, tau):
    """
    Interpolate between two states with given tau (0 to 1).
    Returns single interpolated state.
    NO BATCHING.
    """
    assert q_start.ndim == 1
    assert q_end.ndim == 1
    assert p_start.ndim == 2 
    assert p_end.ndim == 2

    t_start, rot_start = math_utils.unmake_pose(p_start)
    quat_start = math_utils.quat_from_matrix(rot_start)
    t_end, rot_end = math_utils.unmake_pose(p_end)
    quat_end = math_utils.quat_from_matrix(rot_end)
    t_interp = t_start + (t_end - t_start) * tau
    quat_interp = math_utils.quat_slerp(quat_start, quat_end, tau)
    rot_interp = math_utils.matrix_from_quat(quat_interp)
    p_interp = math_utils.make_pose(t_interp, rot_interp)
        
    q_interp = q_start + (q_end - q_start) * tau
    
    return q_interp, p_interp

def find_nearest_neighbor(q_rand, p_rand, dataset_q, dataset_p):
    """
    Find nearest neighbor in dataset.
    NO BATCHING.
    """
    assert q_rand.ndim == 1
    assert p_rand.ndim == 2 
    assert dataset_q.ndim == 2
    assert dataset_p.ndim == 3

    q_rand = q_rand.unsqueeze(0)  # [1, n_dof]
    p_rand = p_rand.unsqueeze(0)  # [1, 4, 4]

    # q distance
    # q_rand: [1, n_dof]
    # dataset_q: [N, n_dof]
    d_q = torch.norm(dataset_q - q_rand, dim=1) # [N]
    
    # p distance
    # Convert to t, q
    t_rand, rot_rand = math_utils.unmake_pose(p_rand) # [1, 3], [1, 3, 3]
    q_rand_quat = math_utils.quat_from_matrix(rot_rand) # [1, 4]
    
    t_data, rot_data = math_utils.unmake_pose(dataset_p) # [N, 3], [N, 3, 3]
    q_data_quat = math_utils.quat_from_matrix(rot_data) # [N, 4]
    
    # Expand to match dataset size
    N = dataset_q.shape[0]
    if t_rand.shape[0] == 1 and N > 1:
        t_rand = t_rand.expand(N, 3) # [N, 3]
        q_rand_quat = q_rand_quat.expand(N, 4) # [N, 4]

    pos_err, rot_err = math_utils.compute_pose_error(
        t_rand, q_rand_quat, t_data, q_data_quat, rot_error_type="axis_angle"
    ) # [N, 3], [N, 3]
    
    d_p_pos = torch.norm(pos_err, dim=1) # [N]
    d_p_rot = torch.norm(rot_err, dim=1) # [N]
    
    # Total distance (weighted)
    # Heuristic weights
    w_q = 1.0
    w_p_pos = 5.0
    w_p_rot = 1.0
    
    d_total = w_q * d_q + w_p_pos * d_p_pos + w_p_rot * d_p_rot # [N]
    
    idx = torch.argmin(d_total) 
    return idx
