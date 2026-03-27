python generate_dataset.py \
--n_grasps=50000 \
--output_dir ./outputs/leap_hand_grasp_cube_v2 \
--push_to_hub iantc104/leap_hand_grasp_cube_v2

python grasp_rrt_expand.py \
--n_grasps=100000 \
--output_dir ./outputs/leap_hand_grasp_cube_v2 \
--push_to_hub iantc104/leap_hand_grasp_cube_v2

python visualize_grasp.py \
--dataset_path ./outputs/leap_hand_grasp_cube_v2