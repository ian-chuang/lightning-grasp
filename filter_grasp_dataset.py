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
    p.add_argument('--batch_size', type=int, default=64, help='Grasps per GPU pass')
    p.add_argument('--limit', type=int, default=0, help='Only check the first N grasps (0 = all)')
    p.add_argument('--output_dir', type=str, default=None, help='Where to write the filtered dataset')
    p.add_argument('--push_to_hub', type=str, default=None, help='Hub repo id to push the result to')
    p.add_argument('--dry_run', action='store_true', help='Measure and report, write nothing')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


class ConvexColliders:
    """Every collision geom of a URDF as a convex polytope in half-space form.

    Faces of all geoms are concatenated into one (A, b) so a whole dataset batch can be
    tested with a single matmul; `face_geom` says which geom each face belongs to.
    """

    def __init__(self, urdf_path, tree, mesh_scale=1.0, device='cuda'):
        robot_mesh = RobotMesh(urdf_path, mesh_scale=mesh_scale)

        A, b, face_geom, link_of_geom, chain_of_geom = [], [], [], [], []
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
                A.append(n)
                b.append(d)
                face_geom.append(np.full(len(n), geom_id))

                verts.append(hull.vertices)
                vert_geom.append(np.full(len(hull.vertices), geom_id))

                link_of_geom.append(link_id)
                chain_of_geom.append(CHAINS.index(chain))

        t = lambda x, dt: torch.as_tensor(np.concatenate(x), dtype=dt, device=device)
        self.A = t(A, torch.float32)                    # [F, 3]
        self.b = t(b, torch.float32)                    # [F]
        self.face_geom = t(face_geom, torch.long)       # [F]
        self.verts = t(verts, torch.float32)            # [V, 3]
        self.vert_geom = t(vert_geom, torch.long)       # [V]
        self.link_of_geom = torch.as_tensor(link_of_geom, dtype=torch.long, device=device)
        self.chain_of_geom = torch.as_tensor(chain_of_geom, dtype=torch.long, device=device)
        self.n_geom = len(link_of_geom)
        self.device = device

    def world_planes(self, geom_pose):
        """Push the half-spaces into the base frame.

        Transforming ~1400 planes per grasp is far cheaper than transforming tens of
        thousands of query points into every geom's local frame.

        Args:
            geom_pose: [B, G, 4, 4]
        Returns:
            A_w: [B, F, 3], b_w: [B, F]
        """
        R = geom_pose[:, self.face_geom, :3, :3]        # [B, F, 3, 3]
        t = geom_pose[:, self.face_geom, :3, 3]         # [B, F, 3]
        A_w = torch.einsum('bfij,fj->bfi', R, self.A)
        b_w = self.b.unsqueeze(0) + (A_w * t).sum(-1)
        return A_w, b_w

    def depth(self, points, geom_pose):
        """Penetration depth of every query point in every geom.

        Args:
            points:    [B, N, 3] in the hand base frame
            geom_pose: [B, G, 4, 4]
        Returns:
            [B, N, G], 0 outside, distance to the nearest face when inside.
        """
        A_w, b_w = self.world_planes(geom_pose)
        s = torch.einsum('bnk,bfk->bnf', points, A_w) - b_w.unsqueeze(1)   # [B, N, F]

        B, N, _ = s.shape
        worst = torch.full((B, N, self.n_geom), -torch.inf, device=s.device)
        worst.scatter_reduce_(
            2, self.face_geom.view(1, 1, -1).expand(B, N, -1), s, reduce='amax', include_self=True
        )
        return worst.neg().clamp_min_(0.0)


def geom_poses(tree, colliders, q):
    return batch_fk(tree, q)["link"][:, colliders.link_of_geom]           # [B, G, 4, 4]


def measure(dataset, robot, args, device='cuda'):
    """Deepest (object, self) penetration per grasp, in metres."""
    tree = build_kinematics_tree(robot.urdf_path, robot.get_active_joints())

    dense_path = args.collision_urdf or robot.urdf_path
    dense = ConvexColliders(dense_path, tree, robot.get_mesh_scale(), device)
    plain = ConvexColliders(robot.urdf_path, tree, robot.get_mesh_scale(), device)
    print(f"  object channel : {os.path.basename(dense_path)}  ({dense.n_geom} colliders)")
    print(f"  self   channel : {os.path.basename(robot.urdf_path)}  ({plain.n_geom} colliders)")

    # A vertex only counts against geoms on another chain: inside one finger,
    # neighbouring links overlap by construction as it flexes.
    cross_chain = plain.chain_of_geom[plain.vert_geom].unsqueeze(1) != plain.chain_of_geom.unsqueeze(0)

    obj_points, _ = MeshObject(args.object_mesh_path).sample_point_and_normal(count=args.n_object_points)
    obj_points = torch.as_tensor(obj_points, dtype=torch.float32, device=device)

    n = len(dataset)
    object_pen = torch.zeros(n, device=device)
    self_pen = torch.zeros(n, device=device)

    from tqdm import tqdm
    for start in tqdm(range(0, n, args.batch_size), desc="Collision"):
        stop = min(start + args.batch_size, n)
        rows = dataset[start:stop]
        q = rows['q'].to(device).float()
        pose = rows['object_pose'].to(device).float()

        # object channel: object surface points against the bone-bridged hand
        P = torch.einsum('bij,nj->bni', pose[:, :3, :3], obj_points) + pose[:, None, :3, 3]
        object_pen[start:stop] = dense.depth(P, geom_poses(tree, dense, q)).amax(dim=(1, 2))

        # self channel: hand collider vertices against other chains' colliders
        gp = geom_poses(tree, plain, q)
        V = torch.einsum('bvij,vj->bvi', gp[:, plain.vert_geom, :3, :3], plain.verts) \
            + gp[:, plain.vert_geom, :3, 3]
        d = plain.depth(V, gp)
        self_pen[start:stop] = (d * cross_chain).amax(dim=(1, 2))

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
