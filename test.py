import torch


def quat_slerp(q1: torch.Tensor, q2: torch.Tensor, tau: float) -> torch.Tensor:
    """Performs spherical linear interpolation (SLERP) between two quaternions.

    This function does not support batch processing.

    Args:
        q1: First quaternion in (w, x, y, z) format.
        q2: Second quaternion in (w, x, y, z) format.
        tau: Interpolation coefficient between 0 (q1) and 1 (q2).

    Returns:
        Interpolated quaternion in (w, x, y, z) format.
    """
    assert isinstance(q1, torch.Tensor), "Input must be a torch tensor"
    assert isinstance(q2, torch.Tensor), "Input must be a torch tensor"
    if tau == 0.0:
        return q1
    elif tau == 1.0:
        return q2
    d = torch.dot(q1, q2)
    if abs(abs(d) - 1.0) < torch.finfo(q1.dtype).eps * 4.0:
        return q1
    if d < 0.0:
        # Invert rotation
        d = -d
        q2 *= -1.0
    angle = torch.acos(torch.clamp(d, -1, 1))
    if abs(angle) < torch.finfo(q1.dtype).eps * 4.0:
        return q1
    isin = 1.0 / torch.sin(angle)
    q1 = q1 * torch.sin((1.0 - tau) * angle) * isin
    q2 = q2 * torch.sin(tau * angle) * isin
    q1 = q1 + q2
    return q1


# ---------------------------------------------------------
# Batched version: q1,q2: (...,4), tau: (...), broadcastable
# ---------------------------------------------------------
def quat_slerp_batched(q1: torch.Tensor, q2: torch.Tensor, tau) -> torch.Tensor:
    """
    Batched SLERP. Accepts (...,4) quaternions and broadcastable tau (...).
    Returns (...,4).
    """
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4

    # Convert tau → tensor
    if not isinstance(tau, torch.Tensor):
        tau = torch.tensor(tau, dtype=q1.dtype, device=q1.device)

    # Broadcast tau to (...,1)
    tau = tau.unsqueeze(-1)  # works for scalar and tensor

    # Dot product over last dim
    d = (q1 * q2).sum(dim=-1, keepdim=True)

    # Handle near-equal quaternions → return q1
    eps = torch.finfo(q1.dtype).eps * 4.0
    close_mask = (abs(abs(d) - 1.0) < eps)

    # Flip q2 when needed
    flip_mask = d < 0
    q2_flipped = torch.where(flip_mask, -q2, q2)
    d = torch.where(flip_mask, -d, d)

    # Angle
    angle = torch.acos(torch.clamp(d, -1.0, 1.0))  # (...,1)

    # Very small angle → return q1
    small_angle_mask = (angle.abs() < eps)

    # Slerp
    sin_angle = torch.sin(angle)
    isin = 1.0 / sin_angle
    part1 = q1 * torch.sin((1.0 - tau) * angle) * isin
    part2 = q2_flipped * torch.sin(tau * angle) * isin
    q = part1 + part2

    # Apply masks: if close or small angle, return q1
    q = torch.where(close_mask | small_angle_mask, q1, q)

    return q


# ---------------------------------------------------------
# Tests (using torch.allclose)
# ---------------------------------------------------------
def test_single_vs_batched():
    torch.manual_seed(0)

    N = 5
    q1 = torch.randn(N, 4)
    q2 = torch.randn(N, 4)
    q1 = q1 / q1.norm(dim=-1, keepdim=True)
    q2 = q2 / q2.norm(dim=-1, keepdim=True)
    taus = torch.rand(N)

    # Individual results
    out_ind = torch.stack([quat_slerp_batched(q1[i], q2[i], float(taus[i])) for i in range(N)])

    # Batched result
    out_batch = quat_slerp_batched(q1, q2, taus)

    print("Individual:")
    print(out_ind)
    print("Batched:")
    print(out_batch)

    # Use torch.allclose
    assert torch.allclose(out_ind, out_batch, atol=1e-6, rtol=1e-6), \
        "Batched output does not match individual SLERP results!"

    print("✓ Test passed: torch.allclose(out_ind, out_batch)")


if __name__ == "__main__":
    test_single_vs_batched()
