import time
import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy
import trimesh
from lygra.robot import build_robot

def main():
    server = viser.ViserServer()
    print("Viser server started")

    # Load Robot
    robot_name = "leap"
    print(f"Loading robot: {robot_name}")
    robot = build_robot(robot_name)
    urdf_path = robot.urdf_path
    
    # Load URDF
    # We use yourdfpy to load it first as seen in previous fixes
    urdf = yourdfpy.URDF.load(urdf_path)
    viser_urdf = ViserUrdf(
        server, 
        urdf_or_path=urdf,
        load_meshes=True,
        load_collision_meshes=False
    )

    # Set all joints to 0
    n_joints = len(viser_urdf.get_actuated_joint_names())
    viser_urdf.update_cfg(np.zeros(n_joints))

    # Get Canonical Space
    box_min, box_max = robot.get_canonical_space()
    
    center = (box_min + box_max) / 2.0
    extents = box_max - box_min
    
    print(f"Canonical Space:")
    print(f"Min: {box_min}")
    print(f"Max: {box_max}")
    print(f"Center: {center}")
    print(f"Extents: {extents}")

    # Create Box Mesh
    # trimesh.creation.box creates a box centered at origin with given extents
    box_mesh = trimesh.creation.box(extents=extents)
    
    # Set color and opacity on the mesh itself
    # RGBA, 0-255
    box_mesh.visual.face_colors = [0, 255, 0, 76] # 0.3 opacity

    # Add Box to Viser
    # We position it at the calculated center
    server.scene.add_mesh_trimesh(
        name="/canonical_space",
        mesh=box_mesh,
        position=center,
    )
    
    # Add a frame at the center of the box for reference
    server.scene.add_frame(
        name="/canonical_space/center",
        position=center,
        axes_length=0.02,
        axes_radius=0.002,
        visible=False,
    )
    
    # Add world axes
    server.scene.add_frame("/world", show_axes=False, axes_length=0.1)

    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    main()
