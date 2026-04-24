```bash

# cube 40mm
uv run python generate_dataset.py \
--robot leap \
--object_mesh_path objects/cube_40mm_m.stl \
--n_grasps=100000 \
--output_dir ./outputs/leap_hand_grasp_dataset_cube_40mm \
--push_to_hub iantc104/leap_hand_grasp_dataset_cube_40mm 

uv run python grasp_rrt_expand.py \
--robot leap \
--object_mesh_path objects/cube_40mm_m.stl \
--n_grasps=200000 \
--dataset_path ./outputs/leap_hand_grasp_dataset_cube_40mm \
--output_dir ./outputs/leap_hand_grasp_dataset_rrt_cube_40mm \
--push_to_hub iantc104/leap_hand_grasp_dataset_rrt_cube_40mm 

# knife
uv run python generate_dataset.py \
--robot leap \
--object_mesh_path objects/knife_m.stl \
--n_grasps=100000 \
--output_dir ./outputs/leap_hand_grasp_dataset_knife \
--push_to_hub iantc104/leap_hand_grasp_dataset_knife 

uv run python grasp_rrt_expand.py \
--robot leap \
--object_mesh_path objects/knife_m.stl \
--n_grasps=200000 \
--dataset_path ./outputs/leap_hand_grasp_dataset_knife \
--output_dir ./outputs/leap_hand_grasp_dataset_rrt_knife \
--push_to_hub iantc104/leap_hand_grasp_dataset_rrt_knife 

# lightbulb
uv run python generate_dataset.py \
--robot leap \
--object_mesh_path objects/lightbulb_m.stl \
--n_grasps=100000 \
--output_dir ./outputs/leap_hand_grasp_dataset_lightbulb \
--push_to_hub iantc104/leap_hand_grasp_dataset_lightbulb

uv run python grasp_rrt_expand.py \
--robot leap \
--object_mesh_path objects/lightbulb_m.stl \
--n_grasps=200000 \
--dataset_path ./outputs/leap_hand_grasp_dataset_lightbulb \
--output_dir ./outputs/leap_hand_grasp_dataset_rrt_lightbulb \
--push_to_hub iantc104/leap_hand_grasp_dataset_rrt_lightbulb    



# wineglass
uv run python generate_dataset.py \
--robot leap \
--object_mesh_path objects/wineglass_m.stl \
--n_grasps=100000 \
--output_dir ./outputs/leap_hand_grasp_dataset_wineglass \
--push_to_hub iantc104/leap_hand_grasp_dataset_wineglass    

uv run python grasp_rrt_expand.py \
--robot leap \
--object_mesh_path objects/wineglass_m.stl \
--n_grasps=200000 \
--dataset_path ./outputs/leap_hand_grasp_dataset_wineglass \
--output_dir ./outputs/leap_hand_grasp_dataset_rrt_wineglass \
--push_to_hub iantc104/leap_hand_grasp_dataset_rrt_wineglass

```


visualize

```bash
uv run python visualize_grasp.py \
--robot leap \
--object_mesh_path objects/cube_40mm_m.stl \
--dataset_path ./outputs/leap_hand_grasp_dataset_rrt_cube_40mm 

uv run python visualize_grasp.py \
--robot leap \
--object_mesh_path objects/knife_m.stl \
--dataset_path ./outputs/leap_hand_grasp_dataset_rrt_knife 

uv run python visualize_grasp.py \
--robot leap \
--object_mesh_path objects/lightbulb_m.stl \
--dataset_path ./outputs/leap_hand_grasp_dataset_rrt_lightbulb

uv run python visualize_grasp.py \
--robot leap \
--object_mesh_path objects/wineglass_m.stl \
--dataset_path ./outputs/leap_hand_grasp_dataset_rrt_wineglass 
```