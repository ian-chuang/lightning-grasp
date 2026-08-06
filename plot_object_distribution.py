"""
Where do the objects actually end up in a grasp dataset?

Plots the distribution of `object_pose` over the dataset, in the hand's base frame,
against the robot's canonical space:

  * position marginals per axis, with the canonical box drawn on
  * XY / XZ / YZ scatter of the object origins, box overlaid
  * orientation coverage: the object's own +X/+Y/+Z axes on a sphere, which shows
    whether the dataset covers all approach directions or only a wedge

The canonical box constrains a sampled *contact point*, not the object origin, and
half the placements come from palm contacts that ignore the box entirely -- so the
origins are expected to spill outside it. What matters is that the cloud is centred
on the box and roughly isotropic in orientation.

    uv run python plot_object_distribution.py --robot hsl_leap \
        --dataset_path ./outputs/hsl_leap_cube_40mm_rrt \
        --output outputs/plots/cube_40mm.png
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lygra.robot import build_robot
from lygra.utils.dataset_utils import load_grasp_dataset

AXES = ('x', 'y', 'z')


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset_path', type=str, required=True,
                   help='Parquet file, save_to_disk directory, or Hub dataset id')
    p.add_argument('--robot', type=str, default='hsl_leap', help='Robot name (for the canonical box)')
    p.add_argument('--output', type=str, default=None, help='PNG to write (default: alongside the dataset)')
    p.add_argument('--max_samples', type=int, default=20000, help='Subsample this many grasps for the plots')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    rng = np.random.default_rng(args.seed)

    dataset = load_grasp_dataset(args.dataset_path)
    n = len(dataset)
    print(f"Loaded {n:,d} grasps from {args.dataset_path}")

    idx = np.arange(n)
    if n > args.max_samples:
        idx = rng.choice(n, size=args.max_samples, replace=False)
        idx.sort()
    poses = np.asarray(dataset.select(idx.tolist())['object_pose'], dtype=np.float64)
    pos = poses[:, :3, 3]
    rot = poses[:, :3, :3]

    box_min, box_max = build_robot(args.robot).get_canonical_space()

    print(f"\nobject origin, hand base frame [m]   (canonical box in brackets)")
    for i, axis in enumerate(AXES):
        p = np.percentile(pos[:, i], [1, 50, 99])
        print(f"  {axis}   p1={p[0]:+.4f}  p50={p[1]:+.4f}  p99={p[2]:+.4f}"
              f"    [{box_min[i]:+.3f}, {box_max[i]:+.3f}]")
    inside = np.all((pos >= box_min) & (pos <= box_max), axis=1)
    print(f"  origins inside the canonical box: {inside.mean() * 100:.1f}%"
          f"   (expected well under 100%: the box constrains a contact point, not the origin)")

    fig = plt.figure(figsize=(16, 9))

    # Row 1: position marginals.
    for i, axis in enumerate(AXES):
        ax = fig.add_subplot(3, 3, i + 1)
        ax.hist(pos[:, i], bins=80, color='0.4')
        ax.axvspan(box_min[i], box_max[i], color='limegreen', alpha=0.25, label='canonical box')
        ax.set_xlabel(f'object {axis} [m]')
        ax.set_ylabel('count')
        if i == 0:
            ax.legend(fontsize=8)

    # Row 2: position scatter, one plane each.
    for k, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        ax = fig.add_subplot(3, 3, 4 + k)
        ax.scatter(pos[:, a], pos[:, b], s=2, alpha=float(np.clip(2000 / max(len(pos), 1), 0.05, 0.9)),
                   c='tab:blue', lw=0)
        ax.add_patch(plt.Rectangle((box_min[a], box_min[b]),
                                   box_max[a] - box_min[a], box_max[b] - box_min[b],
                                   fill=False, ec='limegreen', lw=2, ls='--'))
        ax.set_xlabel(f'{AXES[a]} [m]')
        ax.set_ylabel(f'{AXES[b]} [m]')
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)

    # Row 3: orientation coverage. Each object body axis, plotted as its direction in
    # the hand frame on a lat/lon grid -- a uniform smear means all approaches are covered.
    for i, axis in enumerate(AXES):
        ax = fig.add_subplot(3, 3, 7 + i)
        d = rot[:, :, i]
        lon = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
        lat = np.degrees(np.arcsin(np.clip(d[:, 2], -1, 1)))
        ax.hist2d(lon, lat, bins=(72, 36), range=((-180, 180), (-90, 90)), cmap='magma')
        ax.set_xlabel(f'object +{axis}: longitude [deg]')
        ax.set_ylabel('latitude [deg]')

    fig.suptitle(f'{os.path.basename(args.dataset_path.rstrip("/"))}  --  '
                 f'{len(idx):,d} of {n:,d} grasps, {args.robot} base frame')
    fig.tight_layout()

    out = args.output
    if out is None:
        base = os.path.basename(args.dataset_path.rstrip('/')).replace('/', '_')
        out = os.path.join('outputs', 'plots', f'{base}_object_distribution.png')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.savefig(out, dpi=100)
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
