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
    Supports batched inputs.
    q: [B, n_dof] or [n_dof]
    p: [B, 4, 4] or [4, 4]
    """
    # Ensure batch dimension
    if q_start.ndim == 1:
        q_start = q_start.unsqueeze(0)
        q_end = q_end.unsqueeze(0)
        p_start = p_start.unsqueeze(0)
        p_end = p_end.unsqueeze(0)
        is_batched = False
    else:
        is_batched = True

    B = q_start.shape[0]
    S = num_steps + 2
    
    # Expand inputs
    # [B, ...] -> [S, B, ...] -> [S*B, ...]
    q_start_exp = q_start.unsqueeze(0).expand(S, B, -1).reshape(-1, q_start.shape[-1])
    p_start_exp = p_start.unsqueeze(0).expand(S, B, 4, 4).reshape(-1, 4, 4)
    q_end_exp = q_end.unsqueeze(0).expand(S, B, -1).reshape(-1, q_end.shape[-1])
    p_end_exp = p_end.unsqueeze(0).expand(S, B, 4, 4).reshape(-1, 4, 4)
    
    # Create tau
    t = torch.linspace(0, 1, steps=S, device=q_start.device) # [S]
    # [S] -> [S, B] -> [S*B]
    tau_exp = t.unsqueeze(1).expand(S, B).reshape(-1)
    
    # Call interpolate_state_tau
    q_interp, p_interp = interpolate_state_tau(q_start_exp, p_start_exp, q_end_exp, p_end_exp, tau_exp)
    
    # Reshape back to [S, B, ...]
    q_path = q_interp.reshape(S, B, -1)
    p_path = p_interp.reshape(S, B, 4, 4)
    
    if not is_batched:
        return q_path[:, 0], p_path[:, 0]
    return q_path, p_path

def interpolate_state_tau(q_start, p_start, q_end, p_end, tau):
    """
    Interpolate between two states with given tau (0 to 1).
    Returns single interpolated state.
    Supports batched inputs.
    """
    # Ensure batch dimension
    if q_start.ndim == 1:
        q_start = q_start.unsqueeze(0)
        q_end = q_end.unsqueeze(0)
        p_start = p_start.unsqueeze(0)
        p_end = p_end.unsqueeze(0)
        is_batched = False
    else:
        is_batched = True

    t_start, rot_start = math_utils.unmake_pose(p_start)
    quat_start = math_utils.quat_from_matrix(rot_start)
    t_end, rot_end = math_utils.unmake_pose(p_end)
    quat_end = math_utils.quat_from_matrix(rot_end)
    
    # Handle tau shape for broadcasting
    if isinstance(tau, torch.Tensor):
        # If tau is [B], make it [B, 1] for linear interp
        tau_linear = tau.view(-1, 1)
    else:
        tau_linear = tau

    t_interp = t_start + (t_end - t_start) * tau_linear
    quat_interp = math_utils.quat_slerp(quat_start, quat_end, tau)
    rot_interp = math_utils.matrix_from_quat(quat_interp)
    p_interp = math_utils.make_pose(t_interp, rot_interp)
        
    q_interp = q_start + (q_end - q_start) * tau_linear
    
    if not is_batched:
        return q_interp.squeeze(0), p_interp.squeeze(0)
    return q_interp, p_interp

def find_nearest_neighbor(q_rand, p_rand, dataset_q, dataset_p, w_q=1.0, w_p_pos=5.0, w_p_rot=1.0):
    """
    Find nearest neighbor in dataset.
    Supports batched inputs.
    q_rand: [B, n_dof] or [n_dof]
    p_rand: [B, 4, 4] or [4, 4]
    dataset_q: [N, n_dof]
    dataset_p: [N, 4, 4]
    
    Returns: indices [B] or scalar index
    """
    # Ensure batch dimension
    if q_rand.ndim == 1:
        q_rand = q_rand.unsqueeze(0)
        p_rand = p_rand.unsqueeze(0)
        is_batched = False
    else:
        is_batched = True
        
    B = q_rand.shape[0]
    N = dataset_q.shape[0]
    
    # q distance
    # [B, N]
    d_q = torch.cdist(q_rand, dataset_q)
    
    # p distance
    t_rand, rot_rand = math_utils.unmake_pose(p_rand) # [B, 3], [B, 3, 3]
    q_rand_quat = math_utils.quat_from_matrix(rot_rand) # [B, 4]
    
    t_data, rot_data = math_utils.unmake_pose(dataset_p) # [N, 3], [N, 3, 3]
    q_data_quat = math_utils.quat_from_matrix(rot_data) # [N, 4]
    
    # Expand to compute all pairs using compute_pose_error
    # We need [B*N, ...]
    
    # [B, 1, 3] -> [B, N, 3] -> [B*N, 3]
    t_rand_exp = t_rand.unsqueeze(1).expand(B, N, 3).reshape(-1, 3)
    q_rand_quat_exp = q_rand_quat.unsqueeze(1).expand(B, N, 4).reshape(-1, 4)
    
    # [1, N, 3] -> [B, N, 3] -> [B*N, 3]
    t_data_exp = t_data.unsqueeze(0).expand(B, N, 3).reshape(-1, 3)
    q_data_quat_exp = q_data_quat.unsqueeze(0).expand(B, N, 4).reshape(-1, 4)
    
    pos_err, rot_err = math_utils.compute_pose_error(
        t_rand_exp, q_rand_quat_exp, t_data_exp, q_data_quat_exp, rot_error_type="axis_angle"
    ) # [B*N, 3], [B*N, 3]
    
    d_p_pos = torch.norm(pos_err, dim=1).reshape(B, N) # [B, N]
    d_p_rot = torch.norm(rot_err, dim=1).reshape(B, N) # [B, N]
    
    d_total = w_q * d_q + w_p_pos * d_p_pos + w_p_rot * d_p_rot # [B, N]
    
    indices = torch.argmin(d_total, dim=1) # [B]
    
    if not is_batched:
        return indices[0]
    return indices
