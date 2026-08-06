<center>

# Lightning Grasp
Public repository for the Lightning Grasp system. [paper](https://arxiv.org/abs/2511.07418)

**Thousands of dexterous grasps in seconds.** 
</center>

![](misc/teaser.png)
## System Requirements
- Operating System: Ubuntu 22.04 or 24.04
- GPU: NVIDIA GPU with Pascal, Volta/Turing, Ampere, or Ada Lovelace architecture
- CUDA 12
## Install
Follow these steps to set up the environment and dependencies.


### Step 1. Python Environment
Choose one of the following setup methods.

#### Option 1. Recommended Conda Environment (Stable)
We provide pre-configured Conda environments for different Python/CUDA combinations. For most users with modern NVIDIA GPUs (RTX 20-series and newer), we recommend:
```
conda env create -f conda_env/conda_py39.yml
conda activate lygra
```
Available environments:
- conda_py39.yml: Python 3.9 + PyTorch 2.8.0 (recommended for newer GPUs)
- conda_py38.yml: Python 3.8 + PyTorch 2.4.1 (for legacy systems)

#### Option 2. Lightweight PIP Installation (Minimal)
Not sure if you are tired of setting up a new heavy-weight environment for every new thing. 

If you prefer a minimal setup and already have Python 3.8/3.9 and pytorch installed:
```
uv venv -p 3.9
source .venv/bin/activate
uv pip install torch 
uv pip install open3d==0.19.0 urdfpy==0.0.22 trimesh==4.9.0
uv pip install networkx==3.2.1
uv pip install datasets
```
Done. (Do not merge these pip install into one line)


### Step 2. Setup CUDA Binaries
This release includes pre-compiled CUDA kernel binaries. The CUDA C++ source code will be published in a future release.

**Setup Steps:**
1. Download the appropriate compiled binaries for your Python version from the [Releases page](https://github.com/zhaohengyin/lightning-grasp/releases)
2. Extract the downloaded files into the `lygra/cpp/build/` directory
3. Verify the file structure matches:
```
lygra/cpp/build/lbvh/lbvh.so
lygra/cpp/build/geometry/geometry.so
```

### Step 3. Setup Assets
Download hand and object assts from the [Releases page](https://github.com/zhaohengyin/lightning-grasp/releases), and put them under the ``./assets`` folder. Verify the file structure matches:

```
./assets/hand/...
./assets/object/...
```


## Run

This fork generates grasp datasets for the **`hsl_leap`** hand
(`my_assets/hand/hsl_leap/`, root link `palm`). Five scripts, run in this order:

| | |
|---|---|
| `viz_contact_field.py` | Inspect the contact patches and canonical space on the posed hand. Do this first — everything downstream inherits these settings. |
| `generate_dataset.py` | Synthesise grasps from the contact field. |
| `grasp_rrt_expand.py` | Grow a dataset by RRT expansion from its own rows. |
| `filter_grasp_dataset.py` | Reject deeply-interpenetrating grasps by penetration depth. |
| `visualize_grasp.py` | Browse a dataset, or inspect the k-NN / interpolation mechanism. |
| `plot_object_distribution.py` | Object pose coverage vs the canonical box. |

### Building datasets

`make_datasets.sh` drives generate → expand → filter → plot for any set of objects.
Object names are the mesh basenames in `my_assets/objects/` with `_m.stl` dropped.

```bash
./make_datasets.sh --dry-run                 # print every command, run nothing
./make_datasets.sh                           # all 5 default objects, all 4 stages
./make_datasets.sh cube_40mm knife           # just these two
./make_datasets.sh -v v2 -c 3.0 -n 200000    # new version, tighter box, more grasps
./make_datasets.sh --stages filter,plot      # re-filter datasets that already exist
./make_datasets.sh --no-push --stages gen -n 2000 cube_40mm   # quick local check
./make_datasets.sh --help                    # all options and their defaults
```

A stage whose output directory already exists is skipped, so the script is safe to re-run
after a crash and cheap to extend with a new object. Pass `--force` to redo one. Every
stage tees to `outputs/logs/`, and a failed object doesn't stop the others.

### One-offs

```bash
# inspect the hand config -- contact patches and canonical space, live on the posed hand
uv run python viz_contact_field.py --robot hsl_leap

# browse a dataset, or explore the k-NN / interpolation mechanism (switch modes in the GUI)
uv run python visualize_grasp.py --robot hsl_leap \
  --object_mesh_path my_assets/objects/cube_40mm_m.stl \
  --dataset_path ./outputs/hsl_leap_grasp_cube_40mm_rrt_filtered_v1

# penetration-depth stats and a threshold sweep, writing nothing
uv run python filter_grasp_dataset.py --robot hsl_leap \
  --object_mesh_path my_assets/objects/cube_40mm_m.stl \
  --dataset_path ./outputs/hsl_leap_grasp_cube_40mm_rrt_v1 \
  --collision_urdf my_assets/hand/hsl_leap/urdf/leap_hand_right_dense_collision.urdf \
  --dry_run

# object pose coverage vs the canonical box
uv run python plot_object_distribution.py --robot hsl_leap \
  --dataset_path ./outputs/hsl_leap_grasp_cube_40mm_rrt_filtered_v1
```

Grasps come out in the `palm` frame with joints in `get_active_joints()` order, which for
`hsl_leap` is the hand's own URDF order — no pose transform or joint reindexing downstream.

Object meshes must be in metre units. Supported hands are listed in `lygra/robot/__init__.py`.

### Concentrating grasps toward the middle of the canonical space

A wide canonical space makes goal-conditioned "move the object to a goal pose" harder,
because a random start row and a random goal row can be most of the box apart.

**`--canonical_concentration`** (`-c` on the script) draws each axis from Beta(a, a) rescaled
to the box instead of uniform. `1.0` is uniform and reproduces the original behaviour exactly;
higher pulls toward the centre while keeping full support. On the current box
(half-extents 50/35/41.5 mm):

| a | origin std [mm] | median start→goal | p90 |
|---|---|---|---|
| 1.0 | 29 / 20 / 24 | 56 mm | 85 mm |
| 2.0 (default) | 22 / 16 / 19 | 42 mm | 66 mm |
| 3.0 | 19 / 13 / 16 | 36 mm | 56 mm |

It has to go on **both** stages, which is why `make_datasets.sh` passes it to each: RRT
contributes most of the rows and samples the object *origin* in the box directly, while
generation constrains a contact point instead, so its origins land an object-radius away in
a random direction.

**`--center_sigma`** (`--center-sigma`) does the same thing post-hoc on an existing dataset,
keeping a row at `s` box-radii from the centre with probability `exp(-s²/2σ²)`. Useful for
trying the idea without regenerating — it prints how the start→goal distance moves. It can
only discard rows, so hard concentration costs dataset size.

Neither touches **orientation**, which stays uniform over SO(3) — for a pose-goal task that is
half the difficulty. The stronger fix there is on the RL side: the dataset is an RRT graph, so
rows that are close under `rrt_utils.rank_nearest_neighbors` are reachable from each other by
construction. Sampling the goal from the start's k-nearest neighbours, with k growing over
training, shortens each episode without giving up coverage.

### Other tunables

- **`--n_contact`**: number of active contacts to search during grasp optimization
- **`--batch_size_outer`** & **`--batch_size_inner`**: raise these to fill the GPU.
  Typical steps: (128, 256), (192, 256), (256, 256), (256, 512).
- **`--batch_size`** on the filter trades memory for speed; 64 uses ~3 GB at the default
  8192 object points.

## Setup Your Model
There are several examples in ``lygra/robot/`` folder. You can refer to ``lygra/robot/allegro.py`` for an tutorial. Basically, you simply need to setup a config object that specifies the contact field rules (i.e. which patches to use defined by allowed normals), canonical object space (i.e. where to initialize the object), and some URDF metadata. That's it!

## Notes
**Known Limitations**

This released version has the following limitations.
- The kinematics module does not support mimic joint. The hand DoF must be fully actuated.

- Do not use huge objects for now. I am working on a version that clamps the object mesh and resample the point cloud after placement.

**Comments**

For the grasp synthesis with grippers, consider using specialized approaches for higher efficiency (i.e. directly sample antipodal points on a ray.).


## Troubleshooting:
- If you encounter "Package X not found" errors, simply run ``pip install X``.
-  If you find any error message containing ```networkx```, just run
```
pip install networkx==3.2.1
```

For other problems, leave an issue or email me ``zhaohengyin@cs.berkeley.edu`` (I can be quite busy, but I’ll do my best to respond as soon as I can).
## License
CC-By-NC 4.0. 

[![CC BY-NC 4.0 License](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)


This software and its generated data are licensed for academic and research use only. The license does not grant rights for commercial application, including but not limited to:

- Integration into commercial systems
- Commercial services using this software/data

A separate commercial license is required for any business use. Please contact the author for licensing terms.

## Bibtex
If this work helps your research, a citation would be greatly appreciated!
```
@article{yin2025lightninggrasp,
  title   = {Lightning Grasp: High Performance Procedural Grasp Synthesis with Contact Fields},
  author  = {Yin, Zhao-Heng and Abbeel, Pieter},
  journal = {arXiv preprint arXiv:2511.07418},
  year    = {2025},
  url     = {https://arxiv.org/abs/2511.07418}
}
```
