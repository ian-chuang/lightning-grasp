import torch
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def sample_random_q(robot, batch_size=1):
    """
    Sample random joint configurations within limits.
    """
    tree = robot.get_kinematics_tree()
    lowers = np.array(tree.joint_limit_lowers)
    uppers = np.array(tree.joint_limit_uppers)
    
    # Filter for active joints
    active_joint_ids = [tree.get_joint_id(name) for name in robot.get_active_joints()]
    active_lowers = lowers[active_joint_ids]
    active_uppers = uppers[active_joint_ids]
    
    # Sample
    q_rand = np.random.uniform(active_lowers, active_uppers, size=(batch_size, len(active_joint_ids)))
    return torch.from_numpy(q_rand).float()

def sample_random_object_pose(robot, batch_size=1):
    """
    Sample random object poses (position + orientation).
    Position in canonical space.
    """
    bmin, bmax = robot.get_canonical_space()
    
    # Position
    pos = np.random.uniform(bmin, bmax, size=(batch_size, 3))
    
    # Orientation (Random Quaternion)
    rot = Rotation.random(batch_size)
    
    poses = np.eye(4)[None, ...].repeat(batch_size, axis=0)
    poses[:, :3, 3] = pos
    poses[:, :3, :3] = rot.as_matrix()
    
    return torch.from_numpy(poses).float()

def interpolate_state(q_start, p_start, q_end, p_end, t):
    """
    Interpolate between two states.
    t: float [0, 1]
    """
    # q: Linear
    q_interp = q_start + t * (q_end - q_start)
    
    # p: Position Linear
    pos_start = p_start[:, :3, 3]
    pos_end = p_end[:, :3, 3]
    pos_interp = pos_start + t * (pos_end - pos_start)
    
    # p: Rotation SLERP
    # Convert to numpy for scipy
    if isinstance(p_start, torch.Tensor):
        p_start_np = p_start.detach().cpu().numpy()
    else:
        p_start_np = p_start
        
    if isinstance(p_end, torch.Tensor):
        p_end_np = p_end.detach().cpu().numpy()
    else:
        p_end_np = p_end

    rot_interp_mats = []
    for i in range(len(p_start)):
        r_start = Rotation.from_matrix(p_start_np[i, :3, :3])
        r_end = Rotation.from_matrix(p_end_np[i, :3, :3])
        
        key_rots = Rotation.concatenate([r_start, r_end])
        slerp = Slerp([0, 1], key_rots)
        r_i = slerp([t])
        rot_interp_mats.append(r_i.as_matrix()[0])
        
    rot_interp_mats = np.array(rot_interp_mats)
    
    p_interp = p_start.clone()
    p_interp[:, :3, 3] = pos_interp
    p_interp[:, :3, :3] = torch.from_numpy(rot_interp_mats).float().to(p_start.device)
    
    return q_interp, p_interp

def find_nearest_neighbor(q_rand, p_rand, dataset_q, dataset_p):
    """
    Find nearest neighbor in dataset.
    Simple Euclidean distance on q and p (position).
    TODO: Add rotation distance.
    """
    # q distance
    # q_rand: [1, n_dof]
    # dataset_q: [N, n_dof]
    d_q = torch.norm(dataset_q - q_rand, dim=1)
    
    # p distance (position only for now)
    # p_rand: [1, 4, 4]
    # dataset_p: [N, 4, 4]
    d_p = torch.norm(dataset_p[:, :3, 3] - p_rand[:, :3, 3], dim=1)
    
    # Total distance (weighted)
    # Heuristic weights
    w_q = 1.0
    w_p = 5.0 # Position is important
    
    d_total = w_q * d_q + w_p * d_p
    
    idx = torch.argmin(d_total)
    return idx
