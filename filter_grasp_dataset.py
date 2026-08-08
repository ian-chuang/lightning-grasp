"""
Reject deeply-interpenetrating grasps from a generated dataset.

The generator already runs a collision filter, but it is a *boolean* test against the
plain collision model: an object surface point is either inside a hand link's convex
hull or it is not. Two things slip through.

  1. No depth. A fingertip pressed 1 mm into the object and one buried 30 mm inside it
     both read as "in collision" -- and since a grasp is supposed to touch the object,
     the generator's postprocess has to tolerate the first, which means it tolerates
     the second wherever the boolean test happens to miss.
  2. Gaps between links. The per-link boxes do not meet across a knuckle, so a thin
     object (a knife blade) can thread the gap while barely entering either box.

This script measures penetration *depth* instead, and measures the object channel
against a hand whose knuckle gaps are bridged by extra "bone" colliders -- pass that
URDF as --collision_urdf. Two channels:

  object   deepest penetration of an object surface point into any hand collider.
           Measured on --collision_urdf (bones close the gaps).
  self     deepest penetration of a hand collider vertex into a collider on a
           *different* chain (palm / if / mf / rf / th). Measured on the robot's own
           URDF, NOT the bone-bridged one: bones span joints by construction, so
           neighbouring links overlap in the dense model and its self-collision
           numbers are the hand arguing with itself.

Both run batched on the GPU as pure convex half-space tests; every collider that
lygra builds is already a convex hull, so penetration depth of a point is exactly
-max_f(a_f . p - b_f) over that hull's faces.

    # look first: distributions and a threshold sweep, writes nothing
    uv run python filter_grasp_dataset.py --robot hsl_leap \
        --dataset_path ./outputs/hsl_leap_cube_40mm_rrt \
        --object_mesh_path my_assets/objects/cube_40mm_m.stl \
        --collision_urdf my_assets/hand/hsl_leap/urdf/leap_hand_right_dense_collision.urdf \
        --dry_run

    # then write the filtered dataset
    uv run python filter_grasp_dataset.py ... --output_dir ./outputs/..._filtered
"""

import argparse
import os

import numpy as np
import torch

from lygra.kinematics import batch_fk, build_kinematics_tree
from lygra.mesh import RobotMesh
from lygra.robot import build_robot
from lygra.utils.dataset_utils import load_grasp_dataset
from lygra.utils.geom_utils import MeshObject

CHAINS = ("palm", "if", "mf", "rf", "th")


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset_path', type=str, required=True,
                   help='Parquet file, save_to_disk directory, or Hub dataset id')
    p.add_argument('--robot', type=str, default='hsl_leap', help='Robot name')
    p.add_argument('--object_mesh_path', type=str, required=True, help='Path to the object mesh')
    p.add_argument('--collision_urdf', type=str, default=None,
                   help="URDF for the object channel. Use the bone-bridged one. "
                        "Defaults to the robot's own URDF.")
    p.add_argument('--max_object_penetration', type=float, default=0.008,
                   help='[m] reject above this much hand-inside-object')
    p.add_argument('--max_self_penetration', type=float, default=0.005,
                   help='[m] reject above this much hand-inside-hand, across chains')
    p.add_argument('--center_sigma', type=float, default=0.0,
                   help='Also thin the dataset toward the centre of the canonical space. '
                        'A row at `s` box-radii from the centre is kept with probability '
                        'exp(-s^2 / 2*sigma^2). 0 disables. 1.0 is mild, 0.5 aggressive. '
                        'Unlike --canonical_concentration this needs no regeneration, but it '
                        'can only discard rows, never add them.')
    p.add_argument('--n_object_points', type=int, default=8192,
                   help='Object surface points sampled for the object channel')
    p.add_argument('--batch_size', type=int, default=256,
                   help='Grasps per GPU pass. Peak memory scales with batch_size x '
                        'n_object_points x colliders; drop it if you run out.')
    p.add_argument('--limit', type=int, default=0, help='Only check the first N grasps (0 = all)')
    p.add_argument('--output_dir', type=str, default=None, help='Where to write the filtered dataset')
    p.add_argument('--push_to_hub', type=str, default=None, help='Hub repo id to push the result to')
    p.add_argument('--dry_run', action='store_true', help='Measure and report, write nothing')
    p.add_argument('--device', type=str, default='cuda',
                   help='e.g. cuda:1 to keep off a GPU someone else is training on')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


class ConvexColliders:
    """Every collision geom of a URDF as a convex polytope in half-space form.

    A point p is inside a convex hull iff a_f . p <= b_f for every face f, and its
    penetration depth is then -max_f(a_f . p - b_f). So the whole test is one matmul
    per geom against that geom's plane set.

    Planes are stored per geom rather than concatenated: reducing over the faces of one
    geom at a time is a plain `amax` over the last axis, whereas one concatenated
    (A, b) would need a segmented reduction whose index tensor is larger than the data.
    Coplanar hull triangles are deduplicated first (a box hull is 12 triangles but only
    6 distinct planes), which removes about a third of the work.
    """

    def __init__(self, urdf_path, tree, mesh_scale=1.0, device='cuda'):
        robot_mesh = RobotMesh(urdf_path, mesh_scale=mesh_scale)

        planes, link_of_geom, chain_of_geom = [], [], []
        verts, vert_geom = [], []

        for link_name in tree.get_all_link_names():
            chain = link_name.split('_')[0]
            if chain not in CHAINS:
                raise ValueError(
                    f"link {link_name!r} has prefix {chain!r}, which is not one of {CHAINS}. "
                    "Self-collision is scored per chain, so every link needs one."
                )
            link_id = tree.get_link_id(link_name)

            for mesh in robot_mesh.get_link_collision_meshes_decomposed(link_name):
                if mesh.vertices.shape[0] == 0:
                    continue
                hull = mesh.convex_hull
                geom_id = len(link_of_geom)

                n = hull.face_normals                                     # unit, outward
                d = (n * hull.vertices[hull.faces[:, 0]]).sum(axis=-1)    # a . p = b on the face
                planes.append(np.unique(np.round(np.c_[n, d], 6), axis=0))

                verts.append(hull.vertices)
                vert_geom.append(np.full(len(hull.vertices), geom_id))

                link_of_geom.append(link_id)
                chain_of_geom.append(CHAINS.index(chain))

        as_t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=device)
        self.verts = as_t(np.concatenate(verts), torch.float32)              # [V, 3]
        self.vert_geom = as_t(np.concatenate(vert_geom), torch.long)         # [V]
        self.link_of_geom = as_t(link_of_geom, torch.long)
        self.chain_of_geom = as_t(chain_of_geom, torch.long)
        self.n_geom = len(link_of_geom)
        self.n_face = sum(len(p) for p in planes)
        self.device = device

        # Bounding sphere per geom, for the broad phase.
        centers, radii = [], []
        for p, v in zip(planes, verts):
            c = (v.min(axis=0) + v.max(axis=0)) / 2
            centers.append(c)
            radii.append(np.linalg.norm(v - c, axis=1).max())
        self.center = as_t(np.array(centers), torch.float32)                 # [G, 3]
        self.radius = as_t(np.array(radii), torch.float32)                   # [G]

        # Narrow phase works on a gathered list of (point, geom) candidate pairs, so the
        # plane sets have to be indexable by geom. Geoms are grouped by face count and
        # stacked within a group, which keeps that a plain gather with no padding: here
        # a group is "the 81 boxes" (6 planes each) and "the 4 tip hulls".
        groups = {}
        for g, p in enumerate(planes):
            groups.setdefault(len(p), []).append(g)
        self.groups = []
        self.group_of_geom = torch.zeros(self.n_geom, dtype=torch.long, device=device)
        self.slot_of_geom = torch.zeros(self.n_geom, dtype=torch.long, device=device)
        for i, (n_face, geoms) in enumerate(sorted(groups.items())):
            stacked = np.stack([planes[g] for g in geoms])                   # [Gi, F, 4]
            self.groups.append((as_t(stacked[:, :, :3], torch.float32),
                                as_t(stacked[:, :, 3], torch.float32)))
            self.group_of_geom[as_t(geoms, torch.long)] = i
            self.slot_of_geom[as_t(geoms, torch.long)] = torch.arange(len(geoms), device=device)

    def max_depth(self, points, geom_pose, pair_mask=None):
        """Deepest penetration of any query point into any geom, per batch element.

        Broad phase first: a point can only be inside a geom if it is inside that geom's
        bounding sphere, and that test is one `cdist` producing [B, N, G] floats instead
        of the [B, N, F] the plane test would need. It typically rejects ~99% of pairs,
        and the exact plane test then runs only on the survivors -- which is what makes
        this tractable, since evaluating every plane against every point is ~5x more work
        than the entire rest of the pass.

        Exact, not approximate: the sphere strictly contains the hull, so nothing that
        penetrates can be culled.

        Args:
            points:    [B, N, 3] in the hand base frame
            geom_pose: [B, G, 4, 4]
            pair_mask: [N, G] float, optional. 0 zeroes that (point, geom) pair -- used
                       to ignore same-chain contacts on the self channel.
        Returns:
            [B] metres, 0 where nothing penetrates.
        """
        out = torch.zeros(points.shape[0], device=points.device)

        # --- broad phase
        center_w = torch.einsum('bgij,gj->bgi', geom_pose[:, :, :3, :3], self.center) \
            + geom_pose[:, :, :3, 3]                                         # [B, G, 3]
        near = torch.cdist(points, center_w) < self.radius                   # [B, N, G]
        if pair_mask is not None:
            near &= pair_mask.bool()
        idx = near.nonzero()                                                 # [K, 3] (b, n, g)
        if idx.numel() == 0:
            return out

        # --- narrow phase, on candidates only
        b, n, g = idx[:, 0], idx[:, 1], idx[:, 2]
        R = geom_pose[b, g, :3, :3]                                          # [K, 3, 3]
        local = torch.einsum('kij,kj->ki', R.transpose(1, 2),
                             points[b, n] - geom_pose[b, g, :3, 3])          # [K, 3]

        group = self.group_of_geom[g]
        slot = self.slot_of_geom[g]
        for i, (A, offset) in enumerate(self.groups):
            sel = group == i
            if not sel.any():
                continue
            s = slot[sel]
            depth = (torch.einsum('kfj,kj->kf', A[s], local[sel]) - offset[s]) \
                .amax(-1).neg_().clamp_min_(0.0)                             # [K_i]
            out.scatter_reduce_(0, b[sel], depth, reduce='amax')

        return out


def geom_poses(tree, colliders, q):
    return batch_fk(tree, q)["link"][:, colliders.link_of_geom]           # [B, G, 4, 4]


def measure(dataset, robot, args, device=None):
    device = device or args.device
    """Deepest (object, self) penetration per grasp, in metres."""
    tree = build_kinematics_tree(robot.urdf_path, robot.get_active_joints())

    dense_path = args.collision_urdf or robot.urdf_path
    dense = ConvexColliders(dense_path, tree, robot.get_mesh_scale(), device)
    plain = ConvexColliders(robot.urdf_path, tree, robot.get_mesh_scale(), device)
    print(f"  object channel : {os.path.basename(dense_path)}  "
          f"({dense.n_geom} colliders, {dense.n_face} planes)")
    print(f"  self   channel : {os.path.basename(robot.urdf_path)}  "
          f"({plain.n_geom} colliders, {plain.n_face} planes)")

    # A vertex only counts against geoms on another chain: inside one finger,
    # neighbouring links overlap by construction as it flexes.
    cross_chain = (plain.chain_of_geom[plain.vert_geom].unsqueeze(1)
                   != plain.chain_of_geom.unsqueeze(0)).float()

    obj_points, _ = MeshObject(args.object_mesh_path).sample_point_and_normal(count=args.n_object_points)
    obj_points = torch.as_tensor(obj_points, dtype=torch.float32, device=device)

    n = len(dataset)
    object_pen = torch.zeros(n, device=device)
    self_pen = torch.zeros(n, device=device)

    # Decode the two columns we need once, up front. Slicing the Arrow table per batch
    # costs more than the collision test itself, and at 16 floats a row the whole column
    # is only ~30 MB even at half a million grasps.
    cols = dataset.select_columns(['q', 'object_pose']).with_format('torch')[:]
    all_q = cols['q'].float()
    all_pose = cols['object_pose'].float()

    from tqdm import tqdm
    for start in tqdm(range(0, n, args.batch_size), desc="Collision", unit_scale=args.batch_size,
                      unit="grasp"):
        stop = min(start + args.batch_size, n)
        q = all_q[start:stop].to(device, non_blocking=True)
        pose = all_pose[start:stop].to(device, non_blocking=True)

        # object channel: object surface points against the bone-bridged hand
        P = torch.einsum('bij,nj->bni', pose[:, :3, :3], obj_points) + pose[:, None, :3, 3]
        object_pen[start:stop] = dense.max_depth(P, geom_poses(tree, dense, q))

        # self channel: hand collider vertices against other chains' colliders
        gp = geom_poses(tree, plain, q)
        V = torch.einsum('bvij,vj->bvi', gp[:, plain.vert_geom, :3, :3], plain.verts) \
            + gp[:, plain.vert_geom, :3, 3]
        self_pen[start:stop] = plain.max_depth(V, gp, pair_mask=cross_chain)

    return object_pen.cpu().numpy(), self_pen.cpu().numpy()


def report(object_pen, self_pen, args):
    def q(v):
        p = np.quantile(v, [0.5, 0.9, 0.99, 1.0]) * 1000.0
        return f"p50={p[0]:6.2f}  p90={p[1]:6.2f}  p99={p[2]:6.2f}  max={p[3]:7.2f}"

    print(f"\n  penetration depth [mm]")
    print(f"    object   {q(object_pen)}")
    print(f"    self     {q(self_pen)}")

    sweep = (0.002, 0.003, 0.005, 0.008, 0.012, 0.020)
    print(f"\n  rejected fraction if the threshold were... (the other held at its default)")
    print(f"    {'':10s}" + "".join(f"{t * 1000:>8.0f}mm" for t in sweep))
    for name, values, other, other_max in (
        ("object", object_pen, self_pen, args.max_self_penetration),
        ("self", self_pen, object_pen, args.max_object_penetration),
    ):
        other_bad = other > other_max
        cells = "".join(f"{np.mean((values > t) | other_bad) * 100:9.1f}%" for t in sweep)
        print(f"    {name:10s}{cells}")

    bad_object = object_pen > args.max_object_penetration
    bad_self = self_pen > args.max_self_penetration
    keep = ~(bad_object | bad_self)
    print(f"\n  at object > {args.max_object_penetration * 1000:.0f} mm, "
          f"self > {args.max_self_penetration * 1000:.0f} mm:")
    print(f"    rejected for object : {int(bad_object.sum()):>9,d}  ({bad_object.mean() * 100:5.2f}%)")
    print(f"    rejected for self   : {int(bad_self.sum()):>9,d}  ({bad_self.mean() * 100:5.2f}%)")
    print(f"    rejected for both   : {int((bad_object & bad_self).sum()):>9,d}")
    print(f"    KEPT                : {int(keep.sum()):>9,d}  ({keep.mean() * 100:5.2f}%)")
    return keep


def pair_distance(pos, rng, n=20000):
    """Median / p90 distance between two randomly paired object origins [mm].

    This is the quantity a goal-conditioned policy actually has to cover: how far the
    object must travel between a random start row and a random goal row.
    """
    if len(pos) < 2:
        return float('nan'), float('nan')
    i, j = rng.integers(0, len(pos), n), rng.integers(0, len(pos), n)
    d = np.linalg.norm(pos[i] - pos[j], axis=1) * 1000.0
    return float(np.median(d)), float(np.percentile(d, 90))


def concentrate(dataset, keep, robot, args, rng):
    """Thin the surviving rows toward the centre of the canonical space.

    Acceptance is Gaussian in the object origin's distance from the box centre,
    measured in box radii, so rows outside the box are down-weighted rather than
    hard-rejected (half the placements come from palm contacts, which never
    consulted the box in the first place).
    """
    box_min, box_max = robot.get_canonical_space()
    center = (box_min + box_max) / 2.0
    half = np.maximum((box_max - box_min) / 2.0, 1e-6)

    pos = np.asarray(dataset['object_pose'])[:, :3, 3]
    s2 = (((pos - center) / half) ** 2).sum(axis=1)
    weight = np.exp(-0.5 * s2 / (args.center_sigma ** 2))

    accepted = keep & (rng.random(len(weight)) < weight)

    before = pair_distance(pos[keep], rng)
    after = pair_distance(pos[accepted], rng)
    print(f"\n  centre concentration (sigma = {args.center_sigma}):")
    print(f"    kept                : {int(accepted.sum()):>9,d}  "
          f"({accepted.sum() / max(keep.sum(), 1) * 100:5.2f}% of the collision-filtered rows)")
    print(f"    start->goal distance: p50 {before[0]:5.0f} -> {after[0]:5.0f} mm,"
          f"   p90 {before[1]:5.0f} -> {after[1]:5.0f} mm")
    return accepted


def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_grasp_dataset(args.dataset_path).with_format("torch")
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"Loaded {len(dataset):,d} grasps from {args.dataset_path}")

    robot = build_robot(args.robot)
    object_pen, self_pen = measure(dataset, robot, args)
    keep = report(object_pen, self_pen, args)

    if args.center_sigma > 0:
        keep = concentrate(dataset, keep, robot, args, np.random.default_rng(args.seed))

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return

    filtered = dataset.select(np.where(keep)[0].tolist())
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        filtered.with_format(None).save_to_disk(args.output_dir)
        print(f"\nWrote {len(filtered):,d} grasps to {args.output_dir}")
    if args.push_to_hub:
        print(f"Pushing to {args.push_to_hub}...")
        filtered.push_to_hub(args.push_to_hub, split="train")
    if not args.output_dir and not args.push_to_hub:
        print("\nNothing written (pass --output_dir and/or --push_to_hub, or --dry_run).")


if __name__ == '__main__':
    main()
