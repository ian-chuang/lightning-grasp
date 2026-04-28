```bash

# cube 40mm
uv run python generate_dataset.py \
--robot leap \
--object_mesh_path objects/cube_40mm_m.stl \
--n_grasps=100000 \
--output_dir ./outputs/leap_hand_grasp_dataset_cube_40mm_v3 \
--push_to_hub iantc104/leap_hand_grasp_dataset_cube_40mm_v3 

uv run python grasp_rrt_expand.py \
--robot leap \
--object_mesh_path objects/cube_40mm_m.stl \
--n_grasps=200000 \
--dataset_path ./outputs/leap_hand_grasp_dataset_cube_40mm_v3 \
--output_dir ./outputs/leap_hand_grasp_dataset_rrt_cube_40mm_v3 \
--push_to_hub iantc104/leap_hand_grasp_dataset_rrt_cube_40mm_v3 

# knife
uv run python generate_dataset.py \
--robot leap \
--object_mesh_path objects/knife_m.stl \
--n_grasps=100000 \
--output_dir ./outputs/leap_hand_grasp_dataset_knife_v3 \
--push_to_hub iantc104/leap_hand_grasp_dataset_knife_v3 

uv run python grasp_rrt_expand.py \
--robot leap \
--object_mesh_path objects/knife_m.stl \
--n_grasps=200000 \
--dataset_path ./outputs/leap_hand_grasp_dataset_knife_v3 \
--output_dir ./outputs/leap_hand_grasp_dataset_rrt_knife_v3 \
--push_to_hub iantc104/leap_hand_grasp_dataset_rrt_knife_v3

# lightbulb
uv run python generate_dataset.py \
--robot leap \
--object_mesh_path objects/lightbulb_m.stl \
--n_grasps=100000 \
--output_dir ./outputs/leap_hand_grasp_dataset_lightbulb_v3 \
--push_to_hub iantc104/leap_hand_grasp_dataset_lightbulb_v3

uv run python grasp_rrt_expand.py \
--robot leap \
--object_mesh_path objects/lightbulb_m.stl \
--n_grasps=200000 \
--dataset_path ./outputs/leap_hand_grasp_dataset_lightbulb_v3 \
--output_dir ./outputs/leap_hand_grasp_dataset_rrt_lightbulb_v3 \
--push_to_hub iantc104/leap_hand_grasp_dataset_rrt_lightbulb_v3    



# wineglass
uv run python generate_dataset.py \
--robot leap \
--object_mesh_path objects/wineglass_m.stl \
--n_grasps=100000 \
--output_dir ./outputs/leap_hand_grasp_dataset_wineglass_v3 \
--push_to_hub iantc104/leap_hand_grasp_dataset_wineglass_v3    

uv run python grasp_rrt_expand.py \
--robot leap \
--object_mesh_path objects/wineglass_m.stl \
--n_grasps=200000 \
--dataset_path ./outputs/leap_hand_grasp_dataset_wineglass_v3 \
--output_dir ./outputs/leap_hand_grasp_dataset_rrt_wineglass_v3 \
--push_to_hub iantc104/leap_hand_grasp_dataset_rrt_wineglass_v3

```


visualize

```bash
uv run python visualize_grasp.py \
--robot leap \
--object_mesh_path objects/cube_40mm_m.stl \
--dataset_path ./outputs/leap_hand_grasp_dataset_rrt_cube_40mm_v3

uv run python visualize_grasp.py \
--robot leap \
--object_mesh_path objects/knife_m.stl \
--dataset_path ./outputs/leap_hand_grasp_dataset_rrt_knife_v3

uv run python visualize_grasp.py \
--robot leap \
--object_mesh_path objects/lightbulb_m.stl \
--dataset_path ./outputs/leap_hand_grasp_dataset_rrt_lightbulb_v3

uv run python visualize_grasp.py \
--robot leap \
--object_mesh_path objects/wineglass_m.stl \
--dataset_path ./outputs/leap_hand_grasp_dataset_rrt_wineglass_v3
```